from fastai.tabular.all import *
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data and labels
data_file = 'data.csv'
labels_file = 'labels.csv'
data = pd.read_csv(data_file, header=None)
labels = pd.read_csv(labels_file, header=None).squeeze()

# Preprocess the labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(labels)

# Define the tabular data and dataloaders for the model
splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(data))
to = TabularPandas(data, procs=[Categorify, FillMissing, Normalize],
                   cat_names=None,  # Assuming all features are continuous
                   cont_names=list(data.columns[:-1]),  # All columns except the label
                   y_names='label',
                   splits=splits,
                   y_block=CategoryBlock())

dls = to.dataloaders(bs=64)

# Define the model
class SimpleNN(Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs)
        )

    def forward(self, x_cat, x_cont):
        # x_cat would be the categorical data (if any), and x_cont the continuous data
        # Since you have only continuous data, we only use x_cont
        return self.layers(x_cont)

# The rest of your training and evaluation code remains the same

# Create a learner
num_inputs = 24  # You mentioned having 24 features
num_outputs = len(label_encoder.classes_)
model = SimpleNN(num_inputs, num_outputs)
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

# Train the model
learn.fit_one_cycle(10)  # Number of epochs

# Predictions and Confusion Matrix
preds, targets = learn.get_preds()
predicted_labels = torch.argmax(preds, axis=1)
cm = confusion_matrix(targets, predicted_labels)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.savefig('dnn_plot.jpg', dpi=300)  # Saves the plot as a JPG file
plt.show()
