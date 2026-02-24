import torch, os, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import argparse, json
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer
from datetime import datetime
import wandb
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from accelerate import Accelerator
from torch.utils.data.distributed import DistributedSampler

if torch.mps.is_available():
    ROOT_DIR = "/Users/mrmackamoo/Projects/mechanistic-interpretability"
    DEVICE = "cpu"
    GPU = False
else:
    ROOT_DIR = "/h/120/devan/interp/mechanistic-interpretability" # running on sahitya
    DEVICE = "cuda:4"
    GPU = True

load_dotenv(dotenv_path=f"{ROOT_DIR}/.env")

# -------------------------
# Model
# -------------------------
class SAE(nn.Module):
    def __init__(self, 
        embed_dim: int,
        hidden_dim: int,
    ):
        super(SAE, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Learnable pre-bias
        # self.pre_bias = nn.Parameter(torch.zeros(1, embed_dim))

        # Encoder and Decoder combined
        self.encoder = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, embed_dim, bias=True)

        # Initialize bias vectors to 0
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

        self.relu = nn.ReLU()

        # Kaiming uniform initialization for weights
        # nn.init.kaiming_uniform_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        # nn.init.kaiming_uniform_(self.decoder.weight, mode='fan_in', nonlinearity='linear')
        
        # Initialize decoder weights to random directions with l2 norm of 0.1
        with torch.no_grad():
            random_directions = torch.randn_like(self.decoder.weight)
            normalized_directions = random_directions / (random_directions.norm(dim=0, keepdim=True) + 1e-8)
            self.decoder.weight.copy_(normalized_directions * 0.1)

            # Initialize encoder weights to the transpose of decoder weights
            self.encoder.weight.copy_(self.decoder.weight.T)

        # Normalize decoder to unit columns
        # with torch.no_grad():
        #     self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    def forward(self, x):
        # Subtract pre-bias
        x = x # + self.pre_bias
        # Pass through encoder, activation, and decoder
        codes = self.relu(self.encoder(x))
        reconstructed = self.decoder(codes)
        return reconstructed, codes

# -------------------------
# Training
# -------------------------

def mse_loss_fn(reconstructed, original, codes, lambda_, decoder_weight):
    mse_loss = nn.MSELoss()(reconstructed, original)
    lambda_loss = lambda_ * torch.sum(torch.abs(codes).sum(dim=0) * decoder_weight.norm(dim=0))
    return mse_loss + lambda_loss

def directional_loss_fn(reconstructed, original, codes, lambda_, decoder_weight):
    cosine_sim = nn.functional.cosine_similarity(reconstructed, original, dim=1)
    cosine_loss = 1 - cosine_sim.mean()  # Minimize 1 - mean cosine similarity
    lambda_loss = lambda_ * torch.sum(torch.abs(codes).sum(dim=0) * decoder_weight.norm(dim=0))
    return cosine_loss + lambda_loss

LOSS_FNS = {
    'mse': mse_loss_fn,
    'cos': directional_loss_fn
}

def load_model(
    dir: str = None,
    checkpoint: str = None,
    download: bool = False,
    tmp_dir: str = "./tmp"
) -> SAE:
    if download:
        assert checkpoint, "Checkpoint must be provided when downloading from wandb"
        # Download artifact from wandb and extract to tmp_dir
        artifact = wandb.use_artifact(checkpoint, type="model")
        artifact_dir = artifact.download(root=tmp_dir)
        with open(os.path.join(tmp_dir, "model_name.txt"), "w") as f:
            f.write(checkpoint)
    else:
        # Load from local directory
        artifact_dir = dir
    
    weights_path = os.path.join(artifact_dir, "model.pth")
    hyp_path = os.path.join(artifact_dir, "hyperparams.json")

    # Load hyperparameters
    with open(hyp_path, "r") as f:
        hyperparams = json.load(f)

    model = SAE(
        embed_dim=hyperparams["embed_dim"],
        hidden_dim=hyperparams["hidden_dim"],
    )

    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    if checkpoint:
        print(f"Successfully loaded model: {checkpoint}")
    else:
        print(f"Successfully loaded model from {dir}")
    return model

def train(args: argparse.Namespace, 
    model: SAE, 
    train_dataloader: DataLoader, val_dataloader: DataLoader = None, 
    optimizer: optim.Optimizer = None,
    accelerator: Accelerator = None
):
    if not GPU or accelerator.is_main_process:
        # Create run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{args.save_dir}/{args.run_object.name}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Print number of trainable parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in model: {num_params}")

    # Device setup
    # device = torch.device(DEVICE)
    # print(f"Using device: {device}")
    # args.run_object.log({"device": DEVICE})

    # Move model to device
    # model.to(torch.float32).to(device)
    
    model.train()
    
    num_epochs = args.num_epochs
    global_step = 0

    # print(f"memory allocated before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        code_activity = torch.zeros(args.hidden_dim).to(accelerator.device)
        n_samples = 0
        batch_idx = 0
        for batch_embeddings in train_dataloader:
            # batch_embeddings = batch_embeddings.to(device)

            if batch_embeddings.size(0) == 0:
                continue # empty batch
            
            # Forward pass
            outputs, codes = model(batch_embeddings)

            # [TEMPORARY] track code activity for logging
            with torch.no_grad():
                # we reshape to n_f, batch * seq_len so we can sum activity over all tokens
                code_activity += (codes.reshape(codes.size(2), -1).abs() > 0.0001).float().sum(dim=-1)
                n_samples += batch_embeddings.size(0)
            
            # Compute loss
            loss = args.loss_fn(
                outputs, batch_embeddings, codes=codes, 
                lambda_=args.lambda_, 
                decoder_weight=model.decoder.weight if not GPU else model.module.decoder.weight)

            # Backward pass
            optimizer.zero_grad()

            if GPU:
                accelerator.backward(loss)
            else:
                loss.backward()

            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += batch_embeddings.size(0)
            
            # Log loss to wandb every step
            if not GPU or accelerator.is_main_process: args.run_object.log({"train/loss": loss.item(), "global_step": global_step})
            
            # Print and log every args.logging_steps
            if not GPU or accelerator.is_main_process: 
                if (batch_idx + 1) % args.logging_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                    avg_loss = epoch_loss / num_batches
                    print(f"\tBatch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss:.6f}")
                    args.run_object.log({
                        "train/avg_loss": avg_loss,
                        "epoch": epoch + 1,
                        "batch": batch_idx + 1,
                        "global_step": global_step
                    })

            # Run eval and log every args.val_steps
            if val_dataloader:
                if args.val_steps.isdigit() or args.val_steps == 'epoch':
                    do_eval = False
                    if args.val_steps.isdigit():
                        args.val_steps = int(args.val_steps)
                        if (batch_idx + 1) % args.val_steps == 0: do_eval = True
                    elif (batch_idx + 1) == len(train_dataloader): do_eval = True

                    if do_eval:
                        eval_result = eval(model, val_dataloader, args)
                        print(f"Eval Loss at Epoch {epoch+1}, Batch {batch_idx+1}: {eval_result['loss']:.6f}")
                        log_dict = {
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "global_step": global_step
                        }
                        for k, v in eval_result.items():
                            log_dict[f"val/{k}"] = v
                        args.run_object.log(log_dict)

            batch_idx += 1

        # TODO wait for processes, gather codes_activity properly
        if not GPU or accelerator.is_main_process: 
            accelerator.wait_for_everyone()
            code_activity = accelerator.gather(code_activity)
            n_samples = accelerator.gather(n_samples)
            code_activity = code_activity.sum(dim=0)
            n_samples = n_samples.sum()

        avg_loss = epoch_loss / num_batches
        if not GPU or accelerator.is_main_process: print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.6f} Epoch Loss: {epoch_loss:.6f}")
        if not GPU or accelerator.is_main_process: 
            args.run_object.log({
                # "train/epoch_avg_loss": avg_loss,
                "train/epoch_loss": epoch_loss,
                "epoch": epoch + 1,
                "global_step": global_step,

                # [TEMPORARY] log code activity
                "train/feat_n_active": (code_activity > 0).sum().item(),
                "train/sparsity": (code_activity[code_activity > 0] / n_samples).mean().item()
            })
    
        # Save model and log artifact
        if not GPU or accelerator.is_main_process: 
            if (epoch + 1) % args.save_epochs == 0 or (epoch + 1) == num_epochs:
                fp = f"{output_dir}/model_epoch_{epoch+1}.pth"
                save_model(args, model, fp, push_to_wandb=False)

def eval(model: SAE, val_dataloader: DataLoader, args: argparse.Namespace, log_to_wandb: bool = False) -> dict:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    num_batches = 0

    # for each feature (column) we will track number of times it's activated
    activity_tensor = torch.zeros(model.hidden_dim)

    all_codes = []
    # mse = []
    cos_sim = []

    with torch.no_grad():
        for batch_embeddings, in val_dataloader:
            # batch_embeddings = batch_embeddings.to(device)
            outputs, codes = model(batch_embeddings)
            all_codes.append(codes.detach().cpu())

            non_zero_codes = (codes.abs() > 0.0001).float().sum(dim=1)
            cos_ = F.cosine_similarity(outputs, batch_embeddings)[non_zero_codes > 0]
            cos_sim.extend(cos_.cpu().tolist())

            # mse.extend(F.mse_loss(outputs, batch_embeddings, reduction="none").cpu().tolist())

            loss = args.loss_fn(outputs, batch_embeddings, codes=codes, lambda_=args.lambda_, decoder_weight=model.decoder.weight)
            total_loss += loss.item()
            num_batches += 1

            # compute mean activity of all features
            activity_mask = (codes.abs() > 0.0001).float().sum(dim=0)
            activity_tensor += activity_mask.cpu()
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    all_codes = torch.cat(all_codes, dim=0)
    
    num_val_samples = len(val_dataloader.dataset)

    return {
        "loss": avg_loss,
        "mean_cos_sim": sum(cos_sim) / len(cos_sim),
        "%_zero_samples": (len(all_codes) - len(cos_sim)) / len(all_codes) * 100,
        # "mse": sum(mse) / len(mse),
        "feat_n_active": (activity_tensor > 0).sum().item(),
        "feat_avg_activation": (activity_tensor / num_val_samples)[activity_tensor > 0].mean().item(),  # compute average activation over active features. not dead ones
    }

def get_hyperparams(args: argparse.Namespace) -> dict:
    hyperparams = {
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "lambda": args.lambda_,
        "num_epochs": args.num_epochs,
        "loss_fn": args.loss_fn.__name__,
    }
    return hyperparams

def save_model(args, model: SAE, dir: str, push_to_wandb: bool = True, accelerator: Accelerator = None):
    # save the model as as well as all hyperparams (as json) to recreate it
    os.makedirs(dir, exist_ok=True)
    if accelerator:
        model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), f"{dir}/model.pth")
    hyperparams = get_hyperparams(args)
    with open(f"{dir}/hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Model saved to {dir}/model.pth")

    if push_to_wandb:
        # push this whole directory to wandb as an artifact for the run (args.run_object)
        artifact = wandb.Artifact(name="model", type="model")
        artifact.add_dir(dir)
        args.run_object.log_artifact(artifact)

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help="'cuda:0', 'mps', ..., or 'cuda' for all visible GPUs (accelerate)")
    parser.add_argument("--eval", 
                        # action="store_false", 
                        action="store_true",
                        help="Evaluate the model")
    parser.add_argument("--save_dir", type=str, 
                        default=f"{ROOT_DIR}/understanding_superposition/runs", 
                        help="Directory to save models and logs")
    parser.add_argument("--tmp_dir", type=str,
                        default=f"{ROOT_DIR}/understanding_superposition/tmp", 
                        help="Temporary directory for downloading artifacts")
    parser.add_argument("--wandb_api_key", type=str, 
                        help="Weights & Biases API key for logging")
    parser.add_argument("--data_file", type=str, 
                        default=f"{ROOT_DIR}/understanding_superposition/data/bert_words", 
                        help="Path to save/load embeddings")

    # hyperparameters
    parser.add_argument("--loss_fn", type=str, default='mse', choices=LOSS_FNS.keys(), help=str(list(LOSS_FNS.keys())))
    parser.add_argument("--train_batch_size", type=int, default=10, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=500, help="Validation batch size")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=768*256, help="Hidden dimension in encoder/decoder")
    parser.add_argument("--lambda_", type=float, default=5, help="Sparsity penalty coefficient")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")

    # logging & validation
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every X steps")
    parser.add_argument("--save_epochs", type=int, default=10, help="Save model every X epochs")
    parser.add_argument("--val_steps", type=str, default='epoch', help="Validate every X steps")
    parser.add_argument("--checkpoint", type=str, 
                        # default="mrmackamoo/mechanistic-interpretability/model:v13", 
                        help="Path to model checkpoint for evaluation")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    args.loss_fn = LOSS_FNS[args.loss_fn]

    if GPU:
        accelerator = Accelerator()

    # override
    # args.eval = True
    # args.checkpoint = "runs/moonlit-heartthrob-128_20260215_133409/model_epoch_8.pth"

    if args.device:
        DEVICE = args.device
        if args.device.startswith("cuda"):
            GPU = True

    # Initialize wandb run
    if not GPU or accelerator.is_main_process:
        if args.wandb_api_key is None:
            args.wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=args.wandb_api_key)
        run = wandb.init(
            project="mechanistic-interpretability",
            config=vars(args),
            dir=args.save_dir,
            reinit=True
        )

        args.run_object = run

    # Load the WikiText dataset from Hugging Face
    if not GPU or accelerator.is_main_process: print("Loading Dataset...")

    # Load dataset locally
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    if not GPU or accelerator.is_main_process: print(f"{len(dataset)} training samples")

    # Create sampler for DDP
    # sampler = DistributedSampler(dataset, shuffle=True, seed=42)

    # Set the layer to extract embeddings from
    layer = 3  # Change this to the desired layer

    # Initialize the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    bert_model.eval()  # Set to evaluation mode
    # bert_model.to("cpu")  # Ensure the model runs on CPU

    # Define a collator to tokenize and generate embeddings
    def collate_fn(batch):
        texts = [item["text"] for item in batch if item["text"].strip()]
        if not texts:
            return torch.empty(0, 512, 768)  # Return an empty tensor if no valid texts
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = bert_model(**tokenized)
            embeddings = outputs.hidden_states[layer]  # Extract embeddings from the specified layer
        return embeddings

    # Create DataLoaders for training and validation
    # train_dataset = dataset["train"]
    # val_dataset = dataset["validation"]

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        # shuffle=True,
        collate_fn=collate_fn,
        # sampler=sampler if GPU else None,
        drop_last=True
    )

    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=args.val_batch_size,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    # )

    val_dataloader = None

    # print(f"Loaded {len(train_dataset)} training samples")
    # print(f"Loaded {len(val_dataset)} validation samples")

    if args.checkpoint:
        model = load_model(args.checkpoint, args.tmp_dir)
    else:
        model = SAE(args.embed_dim, args.hidden_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    
    if args.eval:
        eval_result = eval(model, val_dataloader, args)
        print(f"Eval Loss: {eval_result['loss']:.6f}")
    else:
        train(args, model, train_dataloader, val_dataloader, optimizer, accelerator if GPU else None)

    # Finish wandb run
    run.finish()
