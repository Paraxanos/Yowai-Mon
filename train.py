import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Losses ---

def compute_load_balancing_loss(routing_probs, top_k_indices, num_experts):
    """Calculates the auxiliary loss to prevent expert collapse."""
    mean_routing_probs = routing_probs.mean(dim=0)
    mask = F.one_hot(top_k_indices, num_classes=num_experts).float()
    expert_mask = mask.sum(dim=1) 
    tokens_per_expert = expert_mask.mean(dim=0)
    return num_experts * torch.sum(mean_routing_probs * tokens_per_expert)

# --- 2. The Model Wrapper for Two-Stage Training ---

class LandslideMoE_Pipeline(nn.Module):
    def __init__(self, encoder, d_model, num_classes=2):
        super().__init__()
        self.encoder = encoder # Your team's SSMoE Block
        
        # Stage 1 Head: Reconstruct the original patch features
        self.reconstruction_head = nn.Linear(d_model, d_model)
        
        # Stage 2 Head: Binary Classification (Landslide vs No Landslide)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward_pretrain(self, x, mask_ratio=0.3):
        """Stage 1: Self-Supervised Masked Reconstruction"""
        batch_size, seq_len, d_model = x.shape
        
        # Create a random mask
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        
        # Clone input and apply mask (e.g., zero out masked tokens)
        masked_x = x.clone()
        masked_x[mask] = 0.0 
        
        # Pass through encoder (encoder must return features and routing probs)
        encoded_features, routing_probs, top_k_indices = self.encoder(masked_x)
        
        # Predict the original tokens
        reconstructed = self.reconstruction_head(encoded_features)
        
        return reconstructed, x, mask, routing_probs, top_k_indices

    def forward_finetune(self, x):
        """Stage 2: Supervised Classification"""
        # Pass full input through encoder
        encoded_features, routing_probs, top_k_indices = self.encoder(x)
        
        # Pool the sequence (e.g., mean pooling across the patch tokens)
        pooled_features = encoded_features.mean(dim=1)
        
        # Classify
        logits = self.classification_head(pooled_features)
        
        return logits, routing_probs, top_k_indices

# --- 3. The Training Engine ---

def run_hackathon_pipeline(model, pretrain_loader, finetune_loader, optimizer, device):
    model.to(device)
    alpha_aux = 0.01 # Weight for the load balancing loss
    
    # ==========================================
    # STAGE 1: Self-Supervised Pretraining
    # ==========================================
    print("--- Starting Stage 1: Pretraining ---")
    model.train()
    # Hackathon tip: Just run 1-2 epochs to prove the pipeline works
    for epoch in range(2): 
        total_pretrain_loss = 0
        for batch_idx, (inputs, _) in enumerate(pretrain_loader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            recon, orig, mask, probs, indices = model.forward_pretrain(inputs)
            
            # Calculate L1 Loss ONLY on the masked tokens
            l1_loss = F.l1_loss(recon[mask], orig[mask])
            
            # Calculate Auxiliary Loss
            aux_loss = compute_load_balancing_loss(probs, indices, num_experts=4)
            
            # Combine and step
            loss = l1_loss + (alpha_aux * aux_loss)
            loss.backward()
            optimizer.step()
            
            total_pretrain_loss += loss.item()
            
        print(f"Pretrain Epoch {epoch+1} | Loss: {total_pretrain_loss/len(pretrain_loader):.4f}")

    # ==========================================
    # STAGE 2: Supervised Fine-Tuning
    # ==========================================
    print("\n--- Starting Stage 2: Fine-Tuning ---")
    # Freeze the reconstruction head, it's no longer needed
    model.reconstruction_head.requires_grad_(False)
    
    for epoch in range(3):
        total_ft_loss = 0
        correct, total = 0, 0
        
        for batch_idx, (inputs, labels) in enumerate(finetune_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits, probs, indices = model.forward_finetune(inputs)
            
            # Calculate Cross Entropy Loss
            ce_loss = F.cross_entropy(logits, labels)
            
            # Calculate Auxiliary Loss
            aux_loss = compute_load_balancing_loss(probs, indices, num_experts=4)
            
            # Combine and step
            loss = ce_loss + (alpha_aux * aux_loss)
            loss.backward()
            optimizer.step()
            
            # Metrics
            total_ft_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        acc = correct / total
        print(f"Fine-Tune Epoch {epoch+1} | Loss: {total_ft_loss/len(finetune_loader):.4f} | Acc: {acc:.4f}")

    return model

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import os

# --- UPGRADE 1: Checkpointing and F1-Score for Phase 3 ---

def run_phase3_training(model, train_loader, val_loader, optimizer, device, save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    alpha_aux = 0.01 
    best_f1 = 0.0
    
    print("\n--- Starting Phase 3: Supervised Fine-Tuning ---")
    
    for epoch in range(3): # Short epochs for the hackathon
        model.train()
        total_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logits, probs, indices = model.forward_finetune(inputs)
            
            # Weighted Cross Entropy to handle 99% stable / 1% landslide imbalance
            # Adjust weights based on your actual data distribution
            weights = torch.tensor([0.1, 0.9]).to(device) 
            ce_loss = F.cross_entropy(logits, labels, weight=weights)
            aux_loss = compute_load_balancing_loss(probs, indices, num_experts=4)
            
            loss = ce_loss + (alpha_aux * aux_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # --- Validation & Checkpointing ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_logits, _, _ = model.forward_finetune(val_inputs)
                preds = torch.argmax(val_logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(val_labels.numpy())
                
        # Calculate Hackathon-winning metrics
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_prec = precision_score(all_labels, all_preds, zero_division=0)
        val_rec = recall_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        
        # Save the critical output
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), f"{save_dir}/best_eegmoe_model.pt")
            print(">>> Saved New Best Checkpoint <<<")

    return model

# --- UPGRADE 2: Expert Specialization Analysis for Phase 4 ---

# Add save_dir="checkpoints" to the arguments
def run_phase4_evaluation(model, test_loader, device, test_case_name="Puthumala", save_dir="checkpoints"):
    """
    Runs the final test set and extracts the routing behavior to prove 
    Domain-Decoupling worked for the final report.
    """
    print(f"\n--- Starting Phase 4: Evaluating {test_case_name} ---")
    
    # Update this line to use the f-string with save_dir
    model.load_state_dict(torch.load(f"{save_dir}/best_eegmoe_model.pt"))
    model.eval()
    
    # ... rest of the function remains exactly the same ...
    
    all_preds = []
    expert_activation_counts = {0: 0, 1: 0, 2: 0, 3: 0} # Assuming 4 specific experts
    
    with torch.no_grad():
        for inputs, _ in test_loader: # We might not have Y for the pure inference step immediately
            inputs = inputs.to(device)
            
            # Forward pass
            logits, probs, top_k_indices = model.forward_finetune(inputs)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            
            # Analyze Expert Routing (Crucial for the report)
            # top_k_indices shape: (batch_size, seq_len, top_k)
            # Flatten and count which experts were used most for this specific terrain/event
            flat_indices = top_k_indices.view(-1).cpu().numpy()
            unique, counts = np.unique(flat_indices, return_counts=True)
            for u, c in zip(unique, counts):
                expert_activation_counts[u] += c
                
    # Normalize the expert usage into percentages
    total_activations = sum(expert_activation_counts.values())
    print(f"\n[Expert Specialization Analysis for {test_case_name}]")
    for expert_id, count in expert_activation_counts.items():
        percentage = (count / total_activations) * 100 if total_activations > 0 else 0
        print(f"Expert {expert_id}: Handled {percentage:.1f}% of tokens")
        
    # Pass all_preds to Member 4 so they can plot the visual heatmap!
    return all_preds, expert_activation_counts

# =====================================================================
# DUMMY EXECUTION BLOCK - RUN THIS TO TEST YOUR ENGINE LOCALLY
# =====================================================================
if __name__ == "__main__":
    print("Starting Local System Check with Dummy Data...\n")

    # 1. Create a Fake Encoder (Mocking Member 2's work)
    class DummyEncoder(nn.Module):
        def __init__(self, d_model, num_experts):
            super().__init__()
            self.d_model = d_model
            self.num_experts = num_experts

        def forward(self, x):
            batch_size, seq_len, _ = x.shape
            # Fake features out of the encoder
            features = torch.randn(batch_size, seq_len, self.d_model, device=x.device)
            # Fake routing probabilities (Softmax distribution)
            logits = torch.randn(batch_size * seq_len, self.num_experts, device=x.device)
            probs = F.softmax(logits, dim=-1)
            # Fake top-k indices (k=2)
            _, indices = torch.topk(probs, 2, dim=-1)
            return features, probs, indices

    # 2. Setup Dimensions & Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    D_MODEL = 256
    NUM_EXPERTS = 4
    BATCH_SIZE = 16
    SEQ_LEN = 64  # Flattened spatial patches

    # 3. Create Fake DataLoaders (Mocking Member 1's work)
    def get_fake_loader(num_batches=5):
        data = []
        for _ in range(num_batches):
            x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)
            # Simulating landslide imbalance: 95% stable (0), 5% landslide (1)
            y = torch.randint(0, 100, (BATCH_SIZE,))
            y = (y > 95).long() 
            data.append((x, y))
        return data

    dummy_train_loader = get_fake_loader(10)
    dummy_val_loader = get_fake_loader(3)
    dummy_test_loader = get_fake_loader(2)

    # 4. Initialize Your Engine
    mock_encoder = DummyEncoder(d_model=D_MODEL, num_experts=NUM_EXPERTS)
    pipeline = LandslideMoE_Pipeline(encoder=mock_encoder, d_model=D_MODEL)
    
    # We use AdamW standard for Transformer/MoE architectures
    optimizer = torch.optim.AdamW(pipeline.parameters(), lr=1e-3) 

    # 5. Fire it up!
    print(">>> TESTING PHASE 3: TRAINING LOOP <<<")
    try:
        trained_model = run_phase3_training(
            model=pipeline,
            train_loader=dummy_train_loader,
            val_loader=dummy_val_loader,
            optimizer=optimizer,
            device=DEVICE,
            save_dir="test_checkpoints"
        )

        print("\n>>> TESTING PHASE 4: EVALUATION PIPELINE <<<")
        _, stats = run_phase4_evaluation(
            model=trained_model,
            test_loader=dummy_test_loader,
            device=DEVICE,
            test_case_name="Dummy_Wayanad",
            save_dir="test_checkpoints"
        )
        print("\n[SUCCESS] System Check Complete. Your engine is ready for real data.")
        
    except Exception as e:
        print(f"\n[CRITICAL FAILURE] The engine crashed: {e}")