import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class HadamardRepara(torch.nn.Module):
  def __init__(self, num_clusters, response_dim, device):
    super().__init__()
    self.p = torch.nn.Parameter(torch.ones(num_clusters, 1)).to(device)
    self.response_dim = response_dim

  def forward(self, x):
    return x * self.p.repeat_interleave(self.response_dim, dim=0)


class SphericalClustering(nn.Module):
    def __init__(self, num_covariates, response_dim, num_clusters, distribution = "pkbd", min_weight=0.05, device='cpu'):
        super(SphericalClustering, self).__init__()
        self.num_covariates = num_covariates
        self.response_dim = response_dim
        self.num_clusters = num_clusters
        self.min_weight = torch.tensor(min_weight)
        self.distribution = distribution
        self.device = device

        # Linear layer to map covariates X to K cluster embeddings (Cx(d*K))
        self.A = nn.Linear(num_covariates, response_dim * num_clusters, bias=False)

        # Preallocate Pi as the log of uniform probabilities (no need for .to(device))
        self.pi = torch.log(torch.ones(1, num_clusters) / num_clusters).to(device)  # Uniform Pi in log space
        # Preallocate W matrix (no need for .to(device))
        self.W = torch.zeros(1, num_clusters)  # Placeholder for the W matrix
        self.loglik = -1e10

        # Placeholder for the mask
        self.mask = torch.ones(1, num_clusters, dtype=torch.bool).to(device)
        self.mask_dynamic = torch.ones(1, num_clusters, dtype=torch.bool).to(device)

    @property
    def active_components(self):
        return torch.sum(self.mask).item()
    
    @property
    def df(self):
        return self.num_covariates * self.response_dim * self.active_components
    
    def __repr__(self):
        # Custom string representation
        details = f"Mixture of {self.distribution} Distributional Regressions with {self.active_components} components and {self.num_covariates} covariates.\n"
        details += f"Log-likelihood: {self.loglik}\n"
        details += super().__repr__()  # Include the nn.Module details
        return details

    def forward(self, X):
        # Forward pass to map covariates X to embeddings
        N = X.size(0)
        embeddings = self.A(X)  # Shape: Nx(d*K)
        embeddings = embeddings.view(N, self.num_clusters, self.response_dim)  # Shape: NxKxd
        if (self.num_clusters > 1):
            embeddings = embeddings[:, self.mask.squeeze()]
        # Compute mu (mean direction) by normalizing across the last dimension
        norms = torch.norm(embeddings, dim=-1, keepdim=True)  # Shape: NxKx1
        mu = embeddings / norms  # Normalized embeddings: NxKxd
        
        # Compute rho by link transformation norm/(norm+1)
        rho = norms / (norms + 1)  # Shape: NxKx1
        
        return mu, rho

    def log_likelihood(self, mu, rho, Y, distribution):
        # Calculate log likelihood for each cluster
        N, K, d = mu.shape
        Y = Y.unsqueeze(2)  # Shape: Nx1xd
        cross_prod = torch.bmm(mu, Y).squeeze(-1)  # NxKx1 -> NxK
        rho = rho.squeeze(-1)  # NxKx1 -> NxK

        term1 = torch.log(1 - rho ** 2)  # NxK
        term2 = torch.log(1 + rho ** 2 - 2 * rho * cross_prod)  # NxK

        if distribution == "pkbd":
            loglik = term1 - d*term2/2 # Shape: NxK
        elif distribution == "spcauchy":
            loglik = (d-1)*term1 - (d-1)*term2  # Shape: NxK
        else:
            raise ValueError("Model must be 'pkbd' or 'spcauchy'")
        
        return loglik

    def E_step(self, loglik_detached):
        # Perform E-step with the current model parameters
        # Sum log-likelihood with log Pi (since Pi is in log space)
        loglik_with_pi = loglik_detached + self.pi  # Element-wise sum with log Pi vector
        
        # Apply softmax to get W (posterior probabilities) NxK
        self.W = torch.softmax(loglik_with_pi, dim=1)

        # Update Pi by column means of W
        new_pi = torch.mean(self.W, dim=0, keepdim=True)  # Shape: 1xK

        mask2 = (new_pi >= self.min_weight)
        if torch.any(~mask2):
            removed_clusters = (torch.arange(self.num_clusters, device = self.device)+1).unsqueeze(0)[self.mask][~mask2.squeeze()].tolist()
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

        self.loglik = torch.logsumexp(loglik_with_pi, dim = 1).sum().item()

        return removed

    def M_step(self, X, Y, W):
        # Perform full M-step with recalculation of model parameters and multiple optimization steps
        mu, rho = self(X)
        loglik = self.log_likelihood(mu, rho, Y, self.distribution)
        # Perform backward pass based on the current W
        weighted_loglik = loglik * W  # NxK element-wise multiplication
        cluster_loglik = torch.sum(weighted_loglik, dim=0)  # 1xK
        loss = -torch.mean(cluster_loglik)  # Minimize negative log likelihood

        return loss
    
    def _preproc(self, X, Y, N, K, optimizer):
        W = torch.zeros(N, K)
        W.fill_(0.1/(K-1))
        W[:,0] = 0.9 
        perm = torch.stack([torch.randperm(K) for _ in range(N)])
        self.W = torch.gather(W, 1, perm).to(self.device)
        W_colnorm = self.W / (torch.sum(self.W, dim=0, keepdim=True))
        for _ in range(20):
            optimizer.zero_grad()  # Reset gradients
            loss = self.M_step(X, Y, W_colnorm)
            loss.backward()
            optimizer.step()

    
    def fit(self, X, Y, num_epochs=100, num_inner_steps=10, lr = 1e-3, tol = 1e-4, reguralisation = 0, plot = True):
        # Fit the model using EM algorithm
        X = X.to(self.device)
        Y = Y.to(self.device)
        if reguralisation > 0:
            optimizer = optim.AdamW(self.parameters(), lr=lr, weight_decay=reguralisation)
            nn.utils.parametrize.register_parametrization(self.A, 'weight', HadamardRepara(self.num_clusters, self.response_dim, self.device))
        else:    
            optimizer = optim.AdamW(self.parameters(), lr=lr)

        models_loglik_old = -1e10

        Loglikelihoods = []

        self.train()
        self._preproc(X, Y, X.size(0), self.active_components, optimizer)
        for epoch in range(num_epochs):
            # E-step
            optimizer.zero_grad()
            mu, rho = self(X)
            loglik = self.log_likelihood(mu, rho, Y, self.distribution)
            loglik_detached = loglik.detach()  # Detach the log-likelihood before the E-step
            rem = self.E_step(loglik_detached)
            if rem:
                loglik = loglik[:, self.mask_dynamic.squeeze()]
            Loglikelihoods.append(self.loglik)

            if abs(self.loglik - models_loglik_old) < tol*100:
                break
            models_loglik_old = self.loglik

            # M-step
            W_colnorm = self.W / torch.sum(self.W, dim=0, keepdim=True)  # Column normalize W
            weighted_loglik = loglik * W_colnorm  # NxK element-wise multiplication
            cluster_loglik = torch.sum(weighted_loglik, dim=0)  # 1xK
            loss = -torch.mean(cluster_loglik)  # Minimize negative log likelihood
            loss.backward()
            optimizer.step()

            # Perform n-1 more M-steps with re-evaluations
            old_loss = loss.item()
            W_colnorm = self.W / (torch.sum(self.W, dim=0, keepdim=True))  # Column normalize W
            for step in range(num_inner_steps-1):
                optimizer.zero_grad()  # Reset gradients
                loss = self.M_step(X, Y, W_colnorm)
                if abs(loss.item() - old_loss) < tol:
                    print(f'   Inner_step {step + 2}/{num_inner_steps}, Loss: {loss.item()}')
                    break
                loss.backward()
                optimizer.step()  # Update model parameters
                old_loss = loss.item()
            loss = loss.item()

            if (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss}, Log-likelihood: {self.loglik}')
        
        if plot:
            # plot the log-likelihoods over the epochs and return them
            plt.plot(range(epoch+1)[3:], Loglikelihoods[3:])
            plt.xlabel('Epochs')
            plt.ylabel('Log-likelihood')
            plt.title('Log-likelihood over epochs')
        return Loglikelihoods
    
    def get_final_W(self, dataloader):
        """
        Compute the final W matrix (posterior probabilities) for the entire dataset.

        Args:
            dataloader (DataLoader): DataLoader for the dataset (must not shuffle data).

        Returns:
            torch.Tensor: The final W matrix of shape (N, K).
        """
        self.eval()  # Set the model to evaluation mode
        W_list = []
        ll = 0

        with torch.no_grad():
            for batch in dataloader:
                X_batch, Y_batch = batch
                X_batch = X_batch.to(self.device)
                Y_batch = Y_batch.to(self.device)

                # Forward pass
                mu, rho = self(X_batch)
                loglik = self.log_likelihood(mu, rho, Y_batch, self.distribution)
                
                # Compute loglikelihood with current Pi (in log space)
                loglik_with_pi = loglik + self.pi  # self.pi is in log space

                # Compute W for this batch
                W_batch = torch.softmax(loglik_with_pi, dim=1)
                # Final log-likelihood
                ll += torch.logsumexp(loglik_with_pi, dim = 1).sum()
                # Append W_batch to W_list
                W_list.append(W_batch)

        # Concatenate W_list to get full W matrix
        self.loglik = ll
        self.W = torch.cat(W_list, dim=0)  # Shape: (N, K)
        self.pi = torch.log(torch.mean(self.W, dim=0, keepdim=True))  # Update Pi
        return None

    def _preproc_dataloader(self, dataloader, K, optimizer):
        N = len(dataloader.dataset)
        W = torch.zeros(N, K)
        W.fill_(0.1/(K-1))
        W[:,0] = 0.9 
        perm = torch.stack([torch.randperm(K) for _ in range(N)])
        self.W = torch.gather(W, 1, perm)
        W_colnorm = self.W / (torch.sum(self.W, dim=0, keepdim=True))
        for _ in range(20):
            optimizer.zero_grad()  # Reset gradients
            for batch in dataloader:
                X, Y, idx = batch
                X = X.to(self.device)
                Y = Y.to(self.device)
                W_batch = W_colnorm[idx].to(self.device)
                loss = self.M_step(X, Y, W_batch)
                loss.backward()
            optimizer.step()



    def fit_dataloader(self, dataloader, num_epochs=100, num_inner_steps=10, lr = 1e-3, tol = 1e-4, plot = True):
        # Fit the model using EM algorithm
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        models_loglik_old = -1e10
        defined_batch_size = dataloader.batch_size
        Loglikelihoods = []

        self.train()
        self._preproc_dataloader(dataloader, self.active_components, optimizer)
        for epoch in range(num_epochs):
            epoch_loglik = 0
            epoch_loss = 0
            W_list = []

            # E-step
            for batch in dataloader:
                X, Y, idx = batch
                X = X.to(self.device)
                Y = Y.to(self.device)

                with torch.no_grad():
                    mu, rho = self(X)
                    loglik = self.log_likelihood(mu, rho, Y, self.distribution)
                    loglik_with_pi = loglik + self.pi  # Element-wise sum with log Pi vector        
                    W_batch = torch.softmax(loglik_with_pi, dim=1)
                    W_list.append(W_batch)
                    batch_loglik = torch.logsumexp(loglik_with_pi, dim = 1).sum()
                    epoch_loglik += batch_loglik.item()

            # Update Pi by column means of W
            self.W = torch.cat(W_list, dim=0)
            new_pi = torch.mean(self.W, dim=0, keepdim=True)  # Shape: 1xK

            mask2 = (new_pi >= self.min_weight)
            if torch.any(~mask2):
                removed_clusters = (torch.arange(self.num_clusters, device = self.device)+1).unsqueeze(0)[self.mask][~mask2.squeeze()].tolist()
                updated_mask = self.mask.clone()  # Clone the current mask to avoid in-place memory issues
                updated_mask[self.mask] = mask2  # Only update the active part of the original mask
                self.mask = updated_mask
                new_pi = new_pi[:, mask2.squeeze()]
                self.pi = torch.log(new_pi / new_pi.sum(dim=1, keepdim=True))
                self.W = self.W[:, mask2.squeeze()]
                print(f"Clusters {removed_clusters} were removed in this iteration.")
            else:
                self.pi = torch.log(new_pi)

            self.loglik = epoch_loglik
            Loglikelihoods.append(self.loglik)
            if abs(self.loglik - models_loglik_old) < tol*100:
                break
            models_loglik_old = self.loglik 

            # M-step
            W_colnorm = self.W / (torch.sum(self.W, dim=0, keepdim=True))
            for step in range(num_inner_steps):
                old_loss = 1e10
                step_loss = 0
                for batch in dataloader:
                    optimizer.zero_grad()
                    X, Y, idx = batch
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    W_batch = W_colnorm[idx].to(self.device)
                    actual_batch_size = X.size(0)
                    loss_weight = actual_batch_size / defined_batch_size
                    loss = self.M_step(X, Y, W_batch)*loss_weight
                    step_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                if abs(step_loss - old_loss) < tol:
                    print(f'   Inner_step {step + 1}/{num_inner_steps}, Loss: {step_loss}')
                    break
                old_loss = step_loss
            epoch_loss = step_loss    
            if (epoch + 1) % 1 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss}, Log-likelihood: {self.loglik}')    


        if plot:
            # plot the log-likelihoods over the epochs and return them
            plt.plot(range(epoch+1)[3:], Loglikelihoods[3:])
            plt.xlabel('Epochs')
            plt.ylabel('Log-likelihood')
            plt.title('Log-likelihood over epochs')
        return Loglikelihoods
    
    def predict(self, X):
        # Predict the cluster assignments
        self.eval()
        with torch.inference_mode():
            X = X.to(self.device)
            return self(X)

    def predict_and_cluster(self, X, Y):
        # Predict the cluster assignments and return the cluster assignments
        self.eval()
        with torch.inference_mode():
            X = X.to(self.device)
            Y = Y.to(self.device)
            mu, rho = self(X)
            loglik = self.log_likelihood(mu, rho, Y, self.distribution)
            loglik_with_pi = loglik + self.pi  # Element-wise sum with log Pi vector
            posterior = torch.softmax(loglik_with_pi, dim=1)
            return posterior, mu, rho
                  
       
                  
