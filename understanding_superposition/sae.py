"""
Define the model here

x = some word
e = embed(x)
--- model starts here ---
z = encoder(e)  <- experiment with different architectures
s = Bz          <- B is a semi-basis of relevant dimension
d = decoder(s)  <- experiment with different architectures, should probably be symmetric with encoder
--- model ends here ---
loss = MSE(e, d)
"""
import torch, os
import torch.nn as nn
import torch.optim as optim
import argparse, json
from torch.utils.data import DataLoader, TensorDataset
from basis import tao_construction
from transformers import BertModel, BertTokenizer
from datetime import datetime
import wandb
from dotenv import load_dotenv
from tqdm import tqdm

if torch.mps.is_available():
    ROOT_DIR = "/Users/mrmackamoo/Projects/mechanistic-interpretability"
    DEVICE = "mps"  # running on macbook
else:
    ROOT_DIR = "/h/120/devan/interp/mechanistic-interpretability" # running on sahitya
    DEVICE = "cuda"

load_dotenv(dotenv_path=f"{ROOT_DIR}/.env")

# -------------------------
# Model
# -------------------------
class SAE(nn.Module):
    def __init__(self, 
        embed_dim: int,
        hidden_dim: int,
        lambda_: float,
        target_sparsity: float
    ):
        super(SAE, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.lambda_ = lambda_
        self.target_sparsity = target_sparsity

        # Learnable pre-bias
        self.pre_bias = nn.Parameter(torch.zeros(1, embed_dim))

        # Encoder and Decoder combined
        self.encoder = nn.Linear(embed_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, embed_dim, bias=True)
        self.relu = nn.ReLU()

        # Kaiming uniform initialization for weights
        nn.init.kaiming_uniform_(self.encoder.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.decoder.weight, mode='fan_in', nonlinearity='linear')

        def forward(self, x):
            # Subtract pre-bias
            x = x - self.pre_bias
            # Pass through encoder, activation, and decoder
            codes = self.relu(self.encoder(x))
            reconstructed = self.decoder(codes)
            return reconstructed, codes

# -------------------------
# Training
# -------------------------
def construct_model(embed_dim, hidden_dim,) -> SAE:
    model = SAE(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    )

    def project_decoder_grad(grad):
        W = model.decoder.weight.detach()
        W_norm = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        parallel = (grad * W_norm).sum(dim=1, keepdim=True) * W_norm
        return grad - parallel

    model.decoder.weight.register_hook(project_decoder_grad)

    return model

def generate_dataset(
        words: list[str], output_dir: str, layer: int, 
        train_split: float = 0.8, val_split: float = 0.1,
        batch_size: int = 500,):
    
    train_file = os.path.join(output_dir, f"train_{layer}.pt")
    val_file = os.path.join(output_dir, f"val_{layer}.pt")

    if os.path.exists(train_file) and os.path.exists(val_file):
        train_embeddings = torch.load(train_file)
        val_embeddings = torch.load(val_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        bert = BertModel.from_pretrained("bert-base-cased", output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        embeddings = []

        # Tokenize and embed words in batches
        for i in tqdm(range(0, len(words), batch_size), desc="Generating embeddings"):
            batch_words = words[i:i + batch_size]
            
            # Tokenize the batch of words and ensure they are at position 0
            inputs = tokenizer(batch_words, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

            # Get hidden states from BERT
            with torch.no_grad():
                outputs = bert(**inputs)
                hidden_states = outputs.hidden_states[layer]

            # Extract embeddings for the words at position 0
            batch_embeddings = hidden_states[:, 0, :]  # Shape: (batch_size, 768)

            # Stack embeddings for the batch and add to the overall list
            batch_embeddings = torch.stack(batch_embeddings)  # Shape: (batch_size, 768)
            embeddings.append(batch_embeddings)

        # Concatenate all embeddings and split into train/val sets
        embeddings = torch.cat(embeddings, dim=0)  # Shape: (len(words), 768)
        
        # Randomly shuffle and split embeddings into train and val sets
        num_samples = embeddings.size(0)
        indices = torch.randperm(num_samples)
        train_size = int(train_split * num_samples)
        val_size = int(val_split * num_samples)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        train_embeddings = embeddings[train_indices]
        val_embeddings = embeddings[val_indices]

        torch.save(train_embeddings, train_file)
        torch.save(val_embeddings, val_file)

        # Save indices to output_dir/indices.txt
        with open(os.path.join(output_dir, "indices.txt"), "w") as f:
            for idx in indices.tolist():
                f.write(f"{idx}\n")

    train_dataset = TensorDataset(train_embeddings)
    val_dataset = TensorDataset(val_embeddings)
    return train_dataset, val_dataset

def loss_fn(reconstructed, original, codes, lambda_=10):
    # both are shape (B, d)
    mse_loss = nn.MSELoss()(reconstructed, original)
    l1_loss = lambda_ * torch.mean(torch.abs(codes))
    return mse_loss + l1_loss

def load_model(checkpoint: str, tmp_dir = "./tmp", download: bool = True) -> SAE:
    if download:
        # Download artifact from wandb and extract to tmp_dir
        artifact = wandb.use_artifact(checkpoint, type="model")
        artifact_dir = artifact.download(root=tmp_dir)
        with open(os.path.join(tmp_dir, "model_name.txt"), "w") as f:
            f.write(checkpoint)
    else:
        artifact_dir = tmp_dir
        with open(os.path.join(tmp_dir, "model_name.txt"), "r") as f:
            saved_model_name = f.read().strip()
        assert saved_model_name == checkpoint, f"Model name in txt ({saved_model_name}) does not match checkpoint ({checkpoint}), should override download"
    
    weights_path = os.path.join(artifact_dir, "model.pth")
    hyp_path = os.path.join(artifact_dir, "hyperparams.json")

    # Load hyperparameters
    with open(hyp_path, "r") as f:
        hyperparams = json.load(f)

    model = construct_model(
        embed_dim=hyperparams["embed_dim"],
        hidden_dim=hyperparams["hidden_dim"],
    )

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Successfully loaded model: {checkpoint}")
    return model

def train(args: argparse.Namespace, model: SAE, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.save_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in model: {num_params}")

    # Device setup
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    args.run_object.log({"device": DEVICE})

    # Move model to device
    model.to(torch.float32).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    
    num_epochs = args.num_epochs
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_embeddings,) in enumerate(train_dataloader):
            batch_embeddings = batch_embeddings.to(device)
            
            # Forward pass
            codes = model.encoder(batch_embeddings)
            outputs = model.decoder(codes @ model.B)
            
            # Compute loss
            loss = loss_fn(outputs, batch_embeddings, codes=codes, target_sparsity=args.target_sparsity)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Log loss to wandb every step
            args.run_object.log({"train/loss": loss.item(), "global_step": global_step})
            
            # Print and log every args.logging_steps
            if (batch_idx + 1) % args.logging_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                avg_loss = epoch_loss / num_batches
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}, Avg. Loss: {avg_loss:.6f}")
                args.run_object.log({
                    "train/avg_loss": avg_loss,
                    "epoch": epoch + 1,
                    "batch": batch_idx + 1,
                    "global_step": global_step
                })

            # Run eval and log every args.val_steps
            if val_dataloader is not None and ((batch_idx + 1) % args.val_steps == 0 or (batch_idx + 1) == len(train_dataloader)):
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
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.6f} Epoch Loss: {epoch_loss:.6f}")
        args.run_object.log({
            # "train/epoch_avg_loss": avg_loss,
            "train/epoch_loss": epoch_loss,
            "epoch": epoch + 1,
            "global_step": global_step
        })
    
    # Save model and log artifact
    torch.save(model.state_dict(), f"{output_dir}/model.pth")
    print(f"Model saved to {output_dir}/model.pth")
    save_model(args, model, output_dir)

def eval(model: SAE, val_dataloader: DataLoader, args: argparse.Namespace, log_to_wandb: bool = False) -> dict:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    num_batches = 0

    # for each feature (column) we will track number of times it's activated
    activity_tensor = torch.zeros((model.B.shape[0]))

    with torch.no_grad():
        for batch_embeddings, in val_dataloader:
            batch_embeddings = batch_embeddings.to(device)
            codes = model.encoder(batch_embeddings)
            outputs = model.decoder(codes @ model.B)
            loss = loss_fn(outputs, batch_embeddings, codes=codes, target_sparsity=args.target_sparsity)
            total_loss += loss.item()
            num_batches += 1

            # compute mean activity of all features
            activity_mask = (codes.abs() > args.target_sparsity).float().sum(dim=0)
            activity_tensor += activity_mask.cpu()
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    num_val_samples = len(val_dataloader.dataset)
    mean_sparsity = (activity_tensor / num_val_samples).mean().item()
    activity_range = activity_tensor.max().item() - activity_tensor.min().item()


    return {
        "loss": avg_loss,
        "mean_feature_sparsity": mean_sparsity,
        "activity_range": activity_range,
    }

def get_hyperparams(args: argparse.Namespace) -> dict:
    hyperparams = {
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "lambda": args.lambda_,
        "target_sparsity": args.target_sparsity,
        "num_epochs": args.num_epochs,
    }
    return hyperparams

def save_model(args, model: ..., dir: str):
    # save the model as as well as all hyperparams (as json) to recreate it
    os.makedirs(dir, exist_ok=True)
    torch.save(model.state_dict(), f"{dir}/model.pth")
    hyperparams = get_hyperparams(args)
    with open(f"{dir}/hyperparams.json", "w") as f:
        json.dump(hyperparams, f, indent=4)
    print(f"Model saved to {dir}/model.pth")

    # push this whole directory to wandb as an artifact for the run (args.run_object)
    artifact = wandb.Artifact(name="model", type="model")
    artifact.add_dir(dir)
    args.run_object.log_artifact(artifact)

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
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
                        default=f"{ROOT_DIR}/understanding_superposition/data/mpnet2_words.pt", 
                        help="Path to save/load embeddings")

    # hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=1000, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=500, help="Validation batch size")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=768*256, help="Hidden dimension in encoder/decoder")
    parser.add_argument("--lambda_", type=float, default=1e-3, help="Sparsity penalty coefficient")
    parser.add_argument("--num_epochs", type=int, default=350, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--target_sparsity", type=float, default=1e-3, help="Desired Sparsity Level in Activations")

    # logging & validation
    parser.add_argument("--logging_steps", type=int, default=19, help="Log every X steps")
    parser.add_argument("--val_steps", type=int, default=19, help="Validate every X steps")
    parser.add_argument("--save_steps", type=int, default=10, help="Save model every X steps")
    parser.add_argument("--checkpoint", type=str, 
                        # default="mrmackamoo/mechanistic-interpretability/model:v13", 
                        help="Path to model checkpoint for evaluation")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    # Initialize wandb run
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

    with open(f"{ROOT_DIR}/understanding_superposition/data/bert_words.json", "r") as f:
        words = json.load(f)

    train_dataset, val_dataset = generate_dataset(words, args.data_file, train_split=0.8, val_split=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")

    if args.checkpoint:
        model = load_model(args.checkpoint, args.tmp_dir)
    else:
        model = construct_model(args.embed_dim, args.hidden_dim)
    
    if args.eval:
        eval_result = eval(model, val_dataloader, args)
        print(f"Eval Loss: {eval_result['loss']:.6f}")
    else:
        train(args, model, train_dataloader, val_dataloader)

    # Finish wandb run
    run.finish()
