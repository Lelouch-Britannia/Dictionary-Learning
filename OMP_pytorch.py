def OMP(X, y, n_nonzero_coefs=None, tol=None):
    
    #check for cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #convert np array to tensors
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    
    
    #Check the inputs
    assert X.ndim == 2 and y.ndim >= 1
    assert X.shape[0] == y.shape[0]
    assert n_nonzero_coefs is not None or tol is not None
    
    #Intialize some vars
    n_samples, n_features = X.shape
    d = 1
    #if the samples are of d dim
    if y.ndim >= 1:
        d = y.shape[1]
        
    coefs = torch.zeros(n_features, d, device=device) #solution vector
    residual = y.clone().to(device) #current residual
    active_set = [] #list of active features
    
    
    epochs = 0
    #loop until stopping criterion is met
    while True:
        
        correlations = torch.abs(X.T @ residual)
        best_feature = torch.argmax(correlations)
        
        #Add best feature to active set to find it's sparse coefficient
        active_set.append(best_feature)
        
        #solve LSE for active set
        X_active = X[:, active_set]
        
        #Cuda support depending on availablity of GPU for faster computation
        if device == "cuda":
            print(f"Using cuda for least squares\n")
            coef_active, _ = torch.linalg.lstsq(X_active, y, driver='gels')
        else:
            print(f"Using cpu for least squares\n")
            coef_active, _ = torch.linalg.lstsq(X_active, y, driver='gelsd')
        
        
        coef_active = coef_active[:len(active_set)]
        
        
        #Update the solution vector
        coefs[sorted(set(active_set))] = coef_active
        
        #update the residual
        residual = y - X@coefs
        
        #Check the stopping criteria based on n_nonzero_coefs
        if n_nonzero_coefs is not None and len(active_set) >= n_nonzero_coefs:
            break
        #Check the stopping criteria based on tol
        if tol is not None and torch.norm(residual) <= tol:
            break
        # output information after every 100 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Current error: {torch.norm(residual)}, Time elapsed: {time.process_time()} seconds")

        epoch += 1
        
        
        # Print information after every 10 epochs
        if i > 0 and i % 10 == 0:
            pbar.set_postfix({'Residual norm': torch.norm(residual).item(), 'Active set size': len(active_set)})
            print(f"All the soultions: {coef_active}\n")
            print(f"Solution corresponding to active set: {coef_active}\n")
            
    return coefs