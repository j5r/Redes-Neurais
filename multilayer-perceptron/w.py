delta[i] = np.array([1]+self.net.layers[i].dfunc(
    self.net.v[i+1]
).tolist()).reshape(self.net.v[i+1].size+1, 1) * (self.net.layers[i+1].weigths.T @ delta[i+1])
