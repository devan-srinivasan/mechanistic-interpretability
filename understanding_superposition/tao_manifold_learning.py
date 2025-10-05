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
from sentence_transformers import SentenceTransformer
from datetime import datetime
import wandb
from dotenv import load_dotenv

if torch.mps.is_available():
    ROOT_DIR = "/Users/mrmackamoo/Projects/mechanistic-interpretability"
else:
    ROOT_DIR = "~/interp/mechanistic-interpretability" # running on sahitya

load_dotenv(dotenv_path=f"{ROOT_DIR}/.env")

# -------------------------
# Encoder and Decoder
# -------------------------
class Encoder(nn.Module):
    def __init__(self, input_dim, code_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, code_dim))
        # layers.append(nn.Tanh())  # constrain codes to [-1, 1]
        layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, stream_dim, output_dim, hidden_dims):
        super().__init__()
        layers = []
        prev_dim = stream_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev_dim, hd))
            layers.append(nn.ReLU())
            prev_dim = hd
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------
# Autoencoder With Basis
# -------------------------
class AutoencoderWithBasis(nn.Module):
    def __init__(self, encoder, decoder, basis):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer('B', basis.to(torch.float32))
        # print(self.B.shape)

    def forward(self, e):
        z = self.encoder(e)              # (batch_size, latent_dim)
        s = z @ self.B            # (batch_size, n_basis_vectors)
        d = self.decoder(s)              # (batch_size, embed_dim)
        return d

# -------------------------
# Training
# -------------------------

SPARSITY_THRESHOLD = 1e-3
    
def construct_model(
        dim_in: int = 50,
        n: int = 15**2,
        embed_dim: int = 768,
        encoder_hidden_dims: list[int] = [256],
        decoder_hidden_dims: list[int] = [256],
    ):
    # Hyperparameters
    basis = tao_construction(0, dim_in, width=n)

    # TODO [temporary]: splice the basis to first n vectors lol
    # p = int(n**0.5)
    # basis = basis[:p, :] # every p vectors form a subspace, and we have p of these subspaces

    print(basis.shape)

    # Decoder
    basis_dim = basis.shape[1]

    # Model
    encoder = Encoder(input_dim=embed_dim, code_dim=basis.shape[0], hidden_dims=encoder_hidden_dims)
    decoder = Decoder(stream_dim=basis_dim, output_dim=embed_dim, hidden_dims=decoder_hidden_dims)
    model = AutoencoderWithBasis(encoder, decoder, torch.tensor(basis))

    return model

def generate_dataset(words: list[str], output_dir: str, train_split: float = 0.8, val_split: float = 0.1):
    train_file = os.path.join(output_dir, "train.pt")
    val_file = os.path.join(output_dir, "val.pt")

    if os.path.exists(train_file) and os.path.exists(val_file):
        train_embeddings = torch.load(train_file)
        val_embeddings = torch.load(val_file)
    else:
        os.makedirs(output_dir, exist_ok=True)
        mpnet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        embedding_module = mpnet[0].auto_model.embeddings.word_embeddings

        with torch.no_grad():
            # may need to batch this if too large
            tokens = torch.tensor([e[1] for e in mpnet[0].tokenizer(words)['input_ids']])
            embeddings = embedding_module(tokens.to(mpnet.device))
            embeddings = embeddings.to(torch.float32)
        
        # Shuffle and split
        num_samples = embeddings.shape[0]
        indices = torch.randperm(num_samples)
        train_end = int(num_samples * train_split)
        val_end = train_end + int(num_samples * val_split)
        train_embeddings = embeddings[indices[:train_end]]
        val_embeddings = embeddings[indices[train_end:val_end]]
        torch.save(train_embeddings, train_file)
        torch.save(val_embeddings, val_file)

    train_dataset = TensorDataset(train_embeddings)
    val_dataset = TensorDataset(val_embeddings)
    return train_dataset, val_dataset

def loss_fn(reconstructed, original, codes):
    # both are shape (B, d)
    loss = nn.functional.mse_loss(reconstructed, original)
    mean_cosine_sim_loss = 1 - nn.functional.cosine_similarity(reconstructed, original, dim=-1).mean()

    # sparsity loss
    if codes is not None:
        sparsity_loss = (codes.abs() > SPARSITY_THRESHOLD).float().mean()
    else:
        sparsity_loss = 0

    return loss + mean_cosine_sim_loss + sparsity_loss

def load_model(checkpoint: str, tmp_dir = "./tmp", download: bool = True) -> AutoencoderWithBasis:
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
        dim_in=hyperparams["dim_in"],
        n=hyperparams["n"],
        embed_dim=hyperparams["embed_dim"],
        encoder_hidden_dims=hyperparams["encoder_hidden_dims"],
        decoder_hidden_dims=hyperparams["decoder_hidden_dims"],
    )

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    print(f"Successfully loaded model: {checkpoint}")
    return model

def train(args: argparse.Namespace, model: AutoencoderWithBasis, train_dataloader: DataLoader, val_dataloader: DataLoader = None):
    # Create run directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.save_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Print number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in model: {num_params}")

    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
        args.run_object.log({"device": "mps"})
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
        args.run_object.log({"device": "cuda"})
    else:
        device = torch.device("cpu")
        print("Using CPU")
        args.run_object.log({"device": "cpu"})

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
            loss = loss_fn(outputs, batch_embeddings, codes=codes)

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

def eval(model: AutoencoderWithBasis, val_dataloader: DataLoader, args: argparse.Namespace, log_to_wandb: bool = False) -> dict:
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    num_batches = 0
    identity_accuracies = []
    mean_cosine_errors = []

    # for each feature (column) we will track number of times it's activated
    activity_tensor = torch.zeros((model.B.shape[0]))

    with torch.no_grad():
        for batch_embeddings, in val_dataloader:
            batch_embeddings = batch_embeddings.to(device)
            codes = model.encoder(batch_embeddings)
            outputs = model.decoder(codes @ model.B)
            loss = loss_fn(outputs, batch_embeddings, codes=codes)
            total_loss += loss.item()
            num_batches += 1

            # Compute cosine similarity matrix of shape (B, B) using torch.nn.functional.cosine_similarity
            cosine_sim = torch.nn.functional.cosine_similarity(
                outputs.unsqueeze(1),  # (B, 1, d)
                batch_embeddings.unsqueeze(0),  # (1, B, d)
                dim=-1
            )  # (B, B)


            # For each output, find which batch_embedding it is closest to (highest cosine sim)
            preds = cosine_sim.argmax(dim=1)  # (B,)

            # compute the mean error (not squared) between cosine similarity at argmaxed indices (preds) and 1
            mean_cosine_error = (1 - cosine_sim[torch.arange(len(preds)), preds]).mean().item()
            mean_cosine_errors.append(mean_cosine_error)

            identity_accuracy = (preds == torch.arange(len(preds), device=preds.device)).float().mean().item()
            
            identity_accuracies.append(identity_accuracy)

            # compute mean activity of all features
            activity_mask = (codes.abs() > SPARSITY_THRESHOLD).float().sum(dim=0)
            activity_tensor += activity_mask.cpu()
            
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    mean_identity_acc = sum(identity_accuracies) / len(identity_accuracies) if identity_accuracies else 0.0
    mean_cosine_error = sum(mean_cosine_errors) / len(mean_cosine_errors) if mean_cosine_errors else 0.0
    
    num_val_samples = len(val_dataloader.dataset)
    mean_sparsity = (activity_tensor / num_val_samples).mean().item()
    activity_range = activity_tensor.max().item() - activity_tensor.min().item()


    return {
        "loss": avg_loss,
        "mean_cosine_match_acc": mean_identity_acc,
        "mean_cosine_error": mean_cosine_error,
        "mean_feature_sparsity": mean_sparsity,
        "activity_range": activity_range,
    }

def get_hyperparams(args: argparse.Namespace) -> dict:
    hyperparams = {
        "train_batch_size": args.train_batch_size,
        "val_batch_size": args.val_batch_size,
        "dim_in": args.dim_in,
        "n": args.n,
        "embed_dim": args.embed_dim,
        "encoder_hidden_dims": args.encoder_hidden_dims,
        "decoder_hidden_dims": args.decoder_hidden_dims,
        "num_epochs": args.num_epochs,
    }
    return hyperparams

def save_model(args, model: AutoencoderWithBasis, dir: str):
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
    parser.add_argument("--dim_in", type=int, default=50, help="Input dimension")
    parser.add_argument("--n", type=int, default=15**2, help="Number of basis vectors")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--encoder_hidden_dims", type=int, nargs='+', default=[256], help="List of encoder hidden dimensions")
    parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[256], help="List of decoder hidden dimensions")
    parser.add_argument("--num_epochs", type=int, default=350, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=SPARSITY_THRESHOLD, help="Learning rate")
    parser.add_argument("--k", type=int, default=25, help="Top K for accuracy computation")

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

    # TODO [temporary]: override some args for quick testing
    args.dim_in = 760
    args.n = ( int((60 * 768) ** 0.5) + 1 )**2  # 60 is too high for MPS

    # TODO [temporary]: make encoder and decoder shallow for quick testing
    args.encoder_hidden_dims = []
    args.decoder_hidden_dims = []
    args.num_epochs = 600

    # TODO [temporary]: for loading checkpoint lmao
    # args.eval = True
    # args.checkpoint = "mrmackamoo/mechanistic-interpretability/model:v14"

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

    with open(f"{ROOT_DIR}/understanding_superposition/data/mpnet2_words.json", "r") as f:
        words = json.load(f)

    train_dataset, val_dataset = generate_dataset(words, args.data_file, train_split=0.8, val_split=0.1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} training samples")
    print(f"Loaded {len(val_dataset)} validation samples")

    if args.checkpoint:
        model = load_model(args.checkpoint, args.tmp_dir)
    else:
        model = construct_model(
            dim_in=args.dim_in,
            n=args.n,
            embed_dim=args.embed_dim,
            encoder_hidden_dims=args.encoder_hidden_dims,
            decoder_hidden_dims=args.decoder_hidden_dims,
        )
    
    if args.eval:
        eval_result = eval(model, val_dataloader, args)
        print(f"Eval Loss: {eval_result['loss']:.6f}")
    else:
        train(args, model, train_dataloader, val_dataloader)

    # Finish wandb run
    run.finish()
