import torch
import torch.nn as nn
import torch.optim as optim

class PKBDClustering(nn.Module):
    def __init__(self, num_covariates, response_dim, num_clusters, min_weight=0.05, device='cpu'):
        super(PKBDClustering, self).__init__()
        self.num_covariates = num_covariates
        self.response_dim = response_dim
        self.num_clusters = num_clusters
        self.min_weight = torch.tensor(min_weight)
        self.device = device

        # Linear layer to map covariates X to K cluster embeddings (Cx(d*K))
        self.A = nn.Linear(num_covariates, response_dim * num_clusters, bias=False)

        # Preallocate Pi as the log of uniform probabilities (no need for .to(device))
        self.pi = torch.log(torch.ones(1, num_clusters) / num_clusters)  # Uniform Pi in log space
        # Preallocate W matrix (no need for .to(device))
        self.W = torch.zeros(1, num_clusters)  # Placeholder for the W matrix
        self.loglik = -1e10

        # Placeholder for the mask
        self.mask = torch.ones(1, num_clusters, dtype=torch.bool)
        self.mask_dynamic = torch.ones(1, num_clusters, dtype=torch.bool)
        
    def forward(self, X):
        # Forward pass to map covariates X to embeddings
        N = X.size(0)
        embeddings = self.A(X)  # Shape: Nx(d*K)
        embeddings = embeddings.view(N, self.num_clusters, self.response_dim)  # Shape: NxKxd
        embeddings = embeddings[:, self.mask.squeeze()]
        # Compute mu (mean direction) by normalizing across the last dimension
        norms = torch.norm(embeddings, dim=-1, keepdim=True)  # Shape: NxKx1
        mu = embeddings / norms  # Normalized embeddings: NxKxd
        
        # Compute rho by link transformation norm/(norm+1)
        rho = norms / (norms + 1)  # Shape: NxKx1
        
        return mu, rho

    def log_likelihood(self, mu, rho, Y):
        # Calculate log likelihood for each cluster
        N, K, d = mu.shape
        Y = Y.unsqueeze(2)  # Shape: Nx1xd
        cross_prod = torch.bmm(mu, Y).squeeze(-1)  # NxKx1 -> NxK
        rho = rho.squeeze(-1)  # NxKx1 -> NxK

        term1 = torch.log(1 - rho ** 2)  # NxK
        term2 = (d / 2) * torch.log(1 + rho ** 2 - 2 * rho * cross_prod)  # NxK

        loglik = term1 - term2  # Shape: NxK
        return loglik

    def E_step(self, loglik_detached):
        # E-step: update Pi and W based on log likelihood
        N, K = loglik_detached.shape
        # Sum log-likelihood with log Pi (since Pi is in log space)
        loglik_with_pi = loglik_detached + self.pi  # Element-wise sum with log Pi vector
        
        # Apply softmax to get W (posterior probabilities) NxK
        self.W = torch.softmax(loglik_with_pi, dim=1)

        # Update Pi by column means of W
        new_pi = torch.mean(self.W, dim=0, keepdim=True)  # Shape: 1xK

        mask2 = (new_pi >= self.min_weight)
        if torch.any(~mask2):
            removed_clusters = (torch.arange(self.num_clusters)+1).unsqueeze(0)[self.mask][~mask2.squeeze()].tolist()
            updated_mask = self.mask.clone()  # Clone the current mask to avoid in-place memory issues
            updated_mask[self.mask] = mask2  # Only update the active part of the original mask
            self.mask = updated_mask
            self.mask_dynamic = mask2
            loglik_with_pi = loglik_with_pi[:, mask2.squeeze()]
            self.W = torch.softmax(loglik_with_pi, dim=1)
            self.pi = torch.log(torch.mean(self.W, dim=0, keepdim=True))  
            print(f"Clusters {removed_clusters} were removed in this iteration.")
            removed = True
        else:
            self.pi = torch.log(new_pi)
            removed = False

        self.loglik = torch.logsumexp(loglik_with_pi, dim = 1).sum()

        return removed

    def M_step(self, X, Y, optimizer, num_inner_steps=10):
        # Perform full M-step with recalculation of model parameters and multiple optimization steps
        for step in range(num_inner_steps):
            optimizer.zero_grad()  # Reset gradients
            mu, rho = self(X)
            loglik = self.log_likelihood(mu, rho, Y)
            # Perform backward pass based on the current W
            W_colnorm = self.W / (torch.sum(self.W, dim=0, keepdim=True))  # Column normalize W
            weighted_loglik = loglik * W_colnorm  # NxK element-wise multiplication
            cluster_loglik = torch.sum(weighted_loglik, dim=0)  # 1xK
            loss = -torch.mean(cluster_loglik)  # Minimize negative log likelihood
            
            loss.backward()
            optimizer.step()  # Update model parameters
            #if (step + 1) % 3 == 0:
            #    print(f'   Inner_step {step + 2}/{num_inner_steps+1}, Loss: {loss.item()}')
            
        return loss.item()



# Main EM loop
# Main EM loop
def train_em_model(X, Y, model, num_epochs=100, num_inner_steps=10, lr=1e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    X = X.to(model.device)
    Y = Y.to(model.device)

    for epoch in range(num_epochs):
        # E-step
        optimizer.zero_grad()
        mu, rho = model(X)
        loglik = model.log_likelihood(mu, rho, Y)
        loglik_detached = loglik.detach()  # Detach the log-likelihood before the E-step
        rem = model.E_step(loglik_detached)
        if rem:
            loglik = loglik[:, model.mask_dynamic.squeeze()]

        W_colnorm = model.W / torch.sum(model.W, dim=0, keepdim=True)  # Column normalize W
        weighted_loglik = loglik * W_colnorm  # NxK element-wise multiplication
        cluster_loglik = torch.sum(weighted_loglik, dim=0)  # 1xK
        loss = -torch.mean(cluster_loglik)  # Minimize negative log likelihood
        loss.backward()
        optimizer.step()

        # Perform n-1 more M-steps with re-evaluations
        loss = model.M_step(X, Y, optimizer, num_inner_steps=num_inner_steps - 1)

        if (epoch + 1) % 1 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Log-likelihood: {model.loglik}')
       
       
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
N, C, d, K = 100, 2, 1024, 6  # Example dimensions
X = torch.randn(N, C)  # Covariates NxC
Y = torch.randn(N, d)  # Response Nxd
Y = Y / torch.norm(Y, dim=1, keepdim=True)  # Normalize Y

model = PKBDClustering(num_covariates=C, response_dim=d, num_clusters=K, device=device)
model.to(device)  # Move the entire model to the device

train_em_model(X, Y, model, num_epochs=200, num_inner_steps=10, lr=1e-3)