import numpy as np

class PCA:

    def __init__(self, n_components = None):
        self.n_components = n_components
        self.mean = None
        self.eigenvalues = None
        self.eigenvectors = None
        

    def fit(self, X):
        # here X -> [ --img1-- ]      
        #           [ --img2-- ]          
        #           [ --imgi-- ]       
    
        [total_images, image_sz] = X.shape
        
        self.mean = X.mean(axis = 0)
        X = X-self.mean

        if total_images > image_sz :
            cov = np.dot(X.T, X)
            [self.eigenvalues, self.eigenvectors] = np.linalg.eigh(cov)
        else:
            cov = np.dot(X, X.T)
            [self.eigenvalues, self.eigenvectors] = np.linalg.eigh(cov)
            self.eigenvectors = np.dot(X.T, self.eigenvectors)

            for i in range(total_images):
                self.eigenvectors[:,i] = self.eigenvectors[:,i]/np.linalg.norm(self.eigenvectors[:,i])
            

        #sort eigenvalues in descending order
        idx = np.argsort(-self.eigenvalues)
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:,idx]
        
        # select only n_components
        self.eigenvalues = self.eigenvalues[0 : self.n_components].copy()
        self.eigenvectors = self.eigenvectors[ :, 0 : self.n_components].copy()



    def transform(self, X):
        # here X-> [--img--] dim=(no_of_images * image_size)
        if self.mean is None:
            return np.dot(X, self.eigenvectors)

        return np.dot(X - self.mean, self.eigenvectors)
    

    def reconstruct(self, coefficient):
        if self.mean == None:
            return np.dot(coefficient, self.eigenvectors.T)

        return np.dot(coefficient, self.eigenvectors.T) + self.mean

