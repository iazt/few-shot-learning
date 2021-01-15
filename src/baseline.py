class Baseline(nn.Module):
  def __init__(self, nclasses, backbone):
    super(Baseline, self).__init__()
    self.features = backbone()
    self.classifier = nn.Linear(1600, nclasses)
    self.nclasses = nclasses

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores

    def forward_embedding(self,x):
    	return self.feature.forward(x)

    def forward_classifier(self,x):
    	return self.classifier.forward(x)

class Baseline_plus(nn.Module):
  def __init__(self, nclasses, backbone):
    super(Baseline, self).__init__()
    self.features = backbone()
    self.classifier = nn.Linear(1600, nclasses)
    self.nclasses = nclasses

    def forward(self,x):
        out  = self.feature.forward(x)
        scores  = self.classifier.forward(out)
        return scores