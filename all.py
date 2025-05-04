# ------------------------------------------------------SVM--------------------------------------------------------
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
y = (y==0).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

svm = SVC(kernel = 'rbf', C=1, gamma='scale')
svm.fit(x_train, y_train)

y_pred = svm.predict(x_test)

print("SVM Accuracy:", accuracy_score(y_test, y_pred))
# ------------------------------------------------------CSP--------------------------------------------------------
from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_values = 3
x = model.new_int_var(0, num_values-1, "x")
y = model.new_int_var(0, num_values-1, "y")
z = model.new_int_var(0, num_values-1, "z")

model.add(x != y)

solver = cp_model.CpSolver()
status = solver.solve(model)

if status == cp_model.OPTIMAL or cp_model.FEASIBLE:
    print(f"x = {solver.value(x)}")
    print(f"y = {solver.value(y)}")
    print(f"z = {solver.value(z)}")

else:
    print("No solution found")

from ortools.sat.python import cp_model

model = cp_model.CpModel()

num_vals = 3
x = model.new_int_var(0, num_vals - 1, "x")
y = model.new_int_var(0, num_vals - 1, "y")
z = model.new_int_var(0, num_vals - 1, "z")
model.add(x != y)
class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables: list[cp_model.IntVar]):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__solution_count = 0

    def on_solution_callback(self) -> None:
        self.__solution_count += 1
        for v in self.__variables:
            print(f"{v} = {self.value(v)}", end = " ")
        print()

    @property
    def solution_count(self) -> int:
        return self.__solution_count
    
solver = cp_model.CpSolver()
solution_printer= VarArraySolutionPrinter([x, y, z])

solver.parameters.enumerate_all_solutions = True
status = solver.solve(model, solution_printer)

from ortools.sat.python import cp_model

def main() -> None:

    model = cp_model.CpModel()

    var_upper_bound = max(47, 50, 37)
    x = model.new_int_var(0, var_upper_bound, "x")
    y = model.new_int_var(0, var_upper_bound, "y")
    z = model.new_int_var(0, var_upper_bound, "z")

    model.add(2*x + 7*y + 3*z <= 50)
    model.add(3*x - 5*y + 7*z <= 45)

    model.maximize(2*x + 2*y + 3*z)

    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Maximum of objective function: {solver.objective_value}\n")
        print(f"x = {solver.value(x)}")
        print(f"y = {solver.value(y)}")
        print(f"z = {solver.value(z)}")
    else:
        print("No solution found")

    print("Statistics:")
    print(f"status: {solver.status_name(status)}")
    print(f"conflicts: {solver.num_conflicts}")
    print(f"branches: {solver.num_branches}")
    print(f"wall time: {solver.wall_time}")

main()


from ortools.sat.python import cp_model

frequencies = [15, 8, 20]
volumes = [2, 1, 3]
distances = [1, 2, 3]
slot_capacities = [3, 3, 3]

num_products = len(frequencies)
num_slots = len(distances)

model = cp_model.CpModel()

# Flattened list
assign = [
    model.new_bool_var(f"assign_p{p}_s{s}")
    for p in range(num_products)
    for s in range(num_slots)
]

# Constraint: each product in exactly one slot
for p in range(num_products):
    model.add(sum(assign[p * num_slots + s] for s in range(num_slots)) == 1)

# Constraint: slot capacities are not exceeded
for s in range(num_slots):
    model.add(
        sum(assign[p * num_slots + s] * volumes[p] for p in range(num_products))
        <= slot_capacities[s]
    )

# Objective: minimize walking cost
model.minimize(
    sum(assign[p * num_slots + s] * frequencies[p] * distances[s]
        for p in range(num_products) for s in range(num_slots))
)

# Solve
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Assignment:")
    for p in range(num_products):
        for s in range(num_slots):
            if solver.value(assign[p * num_slots + s]):
                print(f"Product {p+1} assigned to Slot {s+1}")
    print(f"Total Walking Cost: {solver.objective_value}")
else:
    print("No feasible solution found.")


from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Regions and colors
regions = ['A', 'B', 'C', 'D']
num_colors = 3  # Red, Green, Blue
color_names = ['Red', 'Green', 'Blue']

# Explicitly create variables for each region
A = model.new_int_var(0, num_colors - 1, "A")
B = model.new_int_var(0, num_colors - 1, "B")
C = model.new_int_var(0, num_colors - 1, "C")
D = model.new_int_var(0, num_colors - 1, "D")

# Adjacency constraints (no two adjacent regions can have the same color)
model.add(A != B)  # A cannot be the same color as B
model.add(A != C)  # A cannot be the same color as C
model.add(B != C)  # B cannot be the same color as C
model.add(C != D)  # C cannot be the same color as D

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output the result
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Map Coloring Solution:")
    # Print each region's color explicitly
    color_index_A = solver.value(A)
    print(f"Region A: {color_names[color_index_A]}")
    
    color_index_B = solver.value(B)
    print(f"Region B: {color_names[color_index_B]}")
    
    color_index_C = solver.value(C)
    print(f"Region C: {color_names[color_index_C]}")
    
    color_index_D = solver.value(D)
    print(f"Region D: {color_names[color_index_D]}")
else:
    print("No solution found.")


from ortools.sat.python import cp_model

# Create the model
model = cp_model.CpModel()

# Job durations and number of jobs
job_durations = [3, 2, 2]
num_jobs = len(job_durations)
horizon = sum(job_durations)  # Maximum total time needed

# Create start time variables for each job
start_0 = model.new_int_var(0, horizon, "start_0")
start_1 = model.new_int_var(0, horizon, "start_1")
start_2 = model.new_int_var(0, horizon, "start_2")

# Create interval variables (for no-overlap constraint)
interval_0 = model.new_interval_var(start_0, job_durations[0], start_0 + job_durations[0], "interval_0")
interval_1 = model.new_interval_var(start_1, job_durations[1], start_1 + job_durations[1], "interval_1")
interval_2 = model.new_interval_var(start_2, job_durations[2], start_2 + job_durations[2], "interval_2")

# Ensure no jobs overlap
model.add_no_overlap([interval_0, interval_1, interval_2])

# Optional: Minimize makespan (when the last job finishes)
makespan = model.new_int_var(0, horizon, "makespan")
model.add(makespan >= start_0 + job_durations[0])
model.add(makespan >= start_1 + job_durations[1])
model.add(makespan >= start_2 + job_durations[2])
model.minimize(makespan)

# Solve the model
solver = cp_model.CpSolver()
status = solver.solve(model)

# Output the result
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Job Schedule:")
    print(f"Job 1 starts at time {solver.value(start_0)}")
    print(f"Job 2 starts at time {solver.value(start_1)}")
    print(f"Job 3 starts at time {solver.value(start_2)}")
    print(f"Total time (makespan): {solver.value(makespan)}")
else:
    print("No solution found.")

# ------------------------------------------------------BAYESIAN NETWORK--------------------------------------------------------
suits = ["Hearts", "Diamonds", "Spades", "Clubs"]
ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "King", "Queen", "Ace", "Jack"]

deck = [(rank, suit) for suit in suits for rank in ranks]

red_cards = [card for card in deck if card[1] in ["Hearts", "Diamonds"]]
p_redCard = len(red_cards)/len(deck)
print(f"Probabilty of drawing a red card: {p_redCard}")

heart_cards = [card for card in red_cards if card[1] == "Hearts"]
p_red_heart = len(heart_cards)/len(red_cards)
print(f"Probabilty of drawing a red card which is a heart: {p_red_heart}")

face_cards = [card for card in deck if card[0] in ["King", "Queen", "Ace"]]
diamond_cards = [card for card in face_cards if card[1] == "Diamonds"]
p_diamond_given_face = len(diamond_cards)/len(face_cards)
print(f"Probability of drawing a diamond given face card: {p_diamond_given_face}")

spade_cards = [card for card in face_cards if card[1] == "Spades"]
queen_cards = [card for card in face_cards if card[0] == "Queen"]

spades_or_queens = set(spade_cards + queen_cards)
p_spades_or_queens_given_faceCards = len(spades_or_queens)/len(face_cards)
print(f"Probability of drawing a spade or a queen given a face card: {p_spades_or_queens_given_faceCards}")


from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork(
    [
        ('Intelligence', "Grade"),
        ('StudyHours', "Grade"),
        ('Difficulty', "Grade"),
        ('Grade', "Pass")
    ]
)

cpd_intelligence = TabularCPD(
    
    variable="Intelligence",
    variable_card=2,
    values=[[0.3],
            [0.7]
            ],
    state_names={'Intelligence': ['High', 'Low']}
)

cpd_studyHours = TabularCPD(
    variable= "StudyHours",
    variable_card=2,
    values=[
        [0.4],
        [0.6]
    ],
    state_names={'StudyHours': ["Insufficient", "Sufficient"]}
)

cpd_difficulty = TabularCPD(
    variable= "Difficulty",
    variable_card=2,
    values=[
        [0.4],
        [0.6]
    ],
    state_names={'Difficulty': ["Hard", "Easy"]}
)

cpd_grade = TabularCPD(
    variable="Grade",
    variable_card= 3,
    values=[
        # A, B, C
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4, 0.3], 
        [0.08, 0.15, 0.2, 0.25, 0.3, 0.3, 0.3, 0.3],  
        [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.3, 0.4],  
    ],
    evidence=["Intelligence", "StudyHours", "Difficulty"],
    evidence_card=[2, 2, 2],
    state_names={
        "Intelligence": ["High", "Low"],
        "StudyHours": ["Insufficient", "Sufficient"],
        "Difficulty": ["Hard", "Easy"],
        "Grade": ["A", "B", "C"]
    }
)

cpd_pass = TabularCPD(
    variable="Pass",
    variable_card= 2,
    values=[
        [0.95, 0.80, 0.50],
        [0.05, 0.20, 0.50]
    ],
    evidence=["Grade"],
    evidence_card=[3],
    state_names={
        "Pass": ["Yes", "No"],
        "Grade": ["A", "B", "C"]
    }
)

model.add_cpds(cpd_intelligence, cpd_studyHours, cpd_difficulty, cpd_grade, cpd_pass)

assert model.check_model(), "Model is incorrect"

inference = VariableElimination(model)

result1 = inference.query(variables=["Pass"], evidence={"StudyHours": "Sufficient", "Difficulty": "Hard"})
print(result1)

result2 = inference.query(variables=["Intelligence"], evidence= {"Pass": "Yes"})
print(result2)


from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork(
    [
        ("Disease", "Fever"),
        ("Disease", "Cough"),
        ("Disease", "Fatigue"),
        ("Disease", "Chills")
    ]
)

cpd_disease = TabularCPD(
    variable="Disease",
    variable_card=2,
    values=[
        [0.3],
        [0.7]
    ],
    state_names={"Disease": ["Flu", "Cold"]}
)

cpd_fever = TabularCPD(
    variable="Fever",
    variable_card=2,
    values=[
        [0.9, 0.5],
        [0.1, 0.5]
    ],
    evidence=["Disease"],
    evidence_card=[2],
    state_names={
            'Fever': ['Yes', 'No'],
            'Disease': ["Flu", "Cold"]
    }
)

cpd_cough = TabularCPD(
    variable= "Cough",
    variable_card=2,
    values=[
        [0.8, 0.6],
        [0.2, 0.4]
    ],
    evidence=["Disease"],
    evidence_card=[2],
    state_names={
        "Disease":["Flu", "Cold"],
        "Cough": ["Yes", "No"]
    }
)
cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.7, 0.3],  
        [0.3, 0.7]   
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Fatigue': ['Yes', 'No'],
        'Disease': ['Flu', 'Cold']
    }
)

cpd_chills = TabularCPD(
    variable='Chills',
    variable_card=2,
    values=[
        [0.6, 0.4],  
        [0.4, 0.6]   
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Chills': ['Yes', 'No'],
        'Disease': ['Flu', 'Cold']
    }
)

model.add_cpds(cpd_chills, cpd_cough, cpd_disease, cpd_fatigue, cpd_fever)

assert model.check_model(), "Model is incorrect"

infer = VariableElimination(model)

print("\nTask 1: P(Disease | Fever=Yes, Cough=Yes)")
result1 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes'}
)
print(result1)

print("\nTask 2: P(Disease | Fever=Yes, Cough=Yes, Chills=Yes)")
result2 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 'Yes', 'Cough': 'Yes', 'Chills': 'Yes'}
)
print(result2)

print("\nTask 3: P(Fatigue = Yes | Disease = Flu)")
result3 = infer.query(
    variables=['Fatigue'],
    evidence={'Disease': 'Flu'}
)
print(result3)



from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Burglary', 'Alarm'),
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

cpd_burglary = TabularCPD(
    variable='Burglary',
    variable_card=2,
    values=[[0.999], [0.001]]
)

cpd_earthquake = TabularCPD(
    variable='Earthquake',
    values=[[0.998], 
            [0.002]],
    variable_card=2,
)

cpd_alarm = TabularCPD(
    variable='Alarm',
    variable_card= 2,
    values=[[0.99, 0.71, 0.06, 0.05],
            [0.001, 0.29, 0.94, 0.95]],
    evidence=['Burglary', 'Earthquake'],
    evidence_card=[2, 2]
)

cpd_johnCalls = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values = [
        [0.3, 0.9],
        [0.7, 0.1]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

cpd_mary = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[
        [0.2, 0.99],
        [0.8, 0.01]
    ],
    evidence=['Alarm'],
    evidence_card=[2]
)

model.add_cpds(cpd_burglary, cpd_earthquake, cpd_johnCalls, cpd_mary, cpd_alarm)

assert model.check_model(), "Model is incorrect"

inference = VariableElimination(model=model)

result = inference.query(variables = ['Burglary'], evidence = {'JohnCalls': 1, 'MaryCalls': 1})
print(result)
# ------------------------------------------------------SUPERVISED--------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

x = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy: .2f}")


# DECISION TREE
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
y = (y == 0).astype(int)
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
random_state=42)
# Create and train the Linear Regression model
LR = LinearRegression()
ModelLR = LR.fit(x_train, y_train)
# Predict on the test data
PredictionLR = ModelLR.predict(x_test)
# Print the predictions
print("Predictions:", PredictionLR)
from sklearn.metrics import r2_score
print('===================LR Testing Accuracy================')
teachLR = r2_score(y_test, PredictionLR)
testingAccLR = teachLR * 100
print(testingAccLR)

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
ModelDt = DT.fit(x_train, y_train)

PredictionDT = DT.predict(x_test)
print("Predictions: ", PredictionDT)

print('====================DT Training Accuracy===============')
tracDT = DT.score(x_train, y_train)
TrainingAccDT = tracDT * 100
print(f"Training Accuracy: {TrainingAccDT:.2f}%")

# Model Testing Accuracy
print('=====================DT Testing Accuracy=================')
teacDT = accuracy_score(y_test, PredictionDT)
testingAccDT = teacDT * 100
print(f"Testing Accuracy: {testingAccDT:.2f}%")
#-----------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and clean the dataset
df = pd.read_excel('customers.xlsx')

# Fill missing values with median
df.fillna({
    'total_spent': df['total_spent'].median(),
    'num_of_visits': df['num_of_visits'].median(),
    'purchase_frequency': df['purchase_frequency'].median()
}, inplace=True)

# Remove outliers using IQR
for col in ['total_spent', 'num_of_visits', 'purchase_frequency']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

# 2. Prepare features and labels
X = df[['total_spent', 'num_of_visits', 'purchase_frequency']]
y = df['value']

# 3. Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 5. Train SVC model
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 6. Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Display hyperplane
weights = model.coef_[0]
bias = model.intercept_[0]
print("\nHyperplane Equation:")
print(f"{weights[0]:.2f}*(total_spent) + {weights[1]:.2f}*(num_of_visits) + {weights[2]:.2f}*(purchase_frequency) + {bias:.2f} = 0")

# 8. Plot decision boundary (2D using total_spent and num_of_visits)
X_2d = X_scaled[['total_spent', 'num_of_visits']]
model_2d = SVC(kernel='linear', C=1.0)
model_2d.fit(X_2d, y)

# Create mesh grid for plotting
x_min, x_max = X_2d['total_spent'].min() - 1, X_2d['total_spent'].max() + 1
y_min, y_max = X_2d['num_of_visits'].min() - 1, X_2d['num_of_visits'].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_2d.predict(grid).reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_2d['total_spent'], X_2d['num_of_visits'], c=y, edgecolors='k', cmap='coolwarm')
plt.xlabel('Total Spent (scaled)')
plt.ylabel('Number of Visits (scaled)')
plt.title('Decision Boundary (SVC - 2D View)')
plt.grid(True)
plt.show()

# 9. Predict new customer
new_customer = pd.DataFrame([[500, 8, 3]], columns=X.columns)
new_customer_scaled = pd.DataFrame(scaler.transform(new_customer), columns=X.columns)
prediction = model.predict(new_customer_scaled)
result = 'HIGH VALUE' if prediction[0] == 1 else 'LOW VALUE'
print(f"\nPrediction for new customer: {result}")







import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

df = pd.read_excel('houses.xlsx')

le = LabelEncoder()
df['neighborhood'] = le.fit_transform(df['neighborhood'])

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

new_house = pd.DataFrame([[2200, 3, 2, 8, le.transform(['A'])[0]]], columns=X.columns)
predicted_price = model.predict(new_house)
print(f"Predicted Price: {predicted_price[0]}")
#-------------------------------------------------------ROC--------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# 1. Generate sample data (replace with your actual data)
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Decision Tree classifier (replace with your trained model if you have one)
DT = DecisionTreeClassifier(max_depth=3, random_state=42)
DT.fit(x_train, y_train)

probabilities = DT.predict_proba(x_test)[:, 1]  # Probability estimates for class 1

# 4. Calculate ROC Curve metrics
fpr, tpr, thresholds = roc_curve(y_test, probabilities)

# 5. Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test, probabilities)

# 6. Plot ROC curve with shaded area under the curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.fill_between(fpr, tpr, color='skyblue', alpha=0.4)  # Shade the area under curve
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with AUC Area')
plt.legend(loc='lower right')
plt.show()
#-------------------------------------------------------TRAIN_TEST_SPLIT----------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

x = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()

model.fit(X=x_train, y=y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy: .2f}")
#------------------------------------------------------KFOLD---------------------------------------------------
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

x = df[['age', 'fare']].fillna(df[['age', 'fare']].mean())
y = df['survived']

x= pd.DataFrame(x)
y= pd.Series(y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression()
accuracy_scores = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)


print("K-Fold CV Average Accuracy: ", np.mean(accuracy_scores))


#-------------------------------------------------LOOCV-----------------------------------------------------------------
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset("titanic")

x = df[['age', 'fare']].fillna(df[['age', 'fare']].mean())
y = df['survived']

x= pd.DataFrame(x)
y= pd.Series(y)

loo = LeaveOneOut()
model = LogisticRegression()
accuracy_scores = []

for train_index, test_index in loo.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_scores.append(acc)


print("K-Fold CV Average Accuracy: ", np.mean(accuracy_scores))
# ------------------------------------------------------UNSUPERVISED--------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'student_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'GPA': [3.8, 2.5, 3.0, 3.2, 2.2, 3.9, 2.8, 3.5, 2.4, 3.7],
    'study_hours': [25, 10, 15, 18, 8, 28, 12, 22, 7, 26],
    'attendance_rate': [92, 65, 75, 80, 60, 95, 70, 85, 55, 90]
}

df = pd.DataFrame(data)
# Select features
features = df.drop(columns=['student_id'])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
wcss = []
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow graph
plt.plot(range(2, 7), wcss, marker='o')
plt.title("Elbow Method to Find Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
# Fit KMeans with optimal K (assume K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

# Add cluster labels to original DataFrame
df['cluster'] = cluster_labels
# Plot using GPA vs study_hours and color by cluster
plt.figure(figsize=(8, 6))
for i in range(optimal_k):
    cluster_data = df[df['cluster'] == i]
    plt.scatter(cluster_data['study_hours'], cluster_data['GPA'], label=f'Cluster {i}')

plt.title("Student Clustering (GPA vs Study Hours)")
plt.xlabel("Study Hours per Week")
plt.ylabel("GPA")
plt.legend()
plt.grid(True)
plt.show()
print("Final Clustered Data:")
print(df[['student_id', 'GPA', 'study_hours', 'attendance_rate', 'cluster']])
# ------------------------------------------------------MATPLOTLIB--------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = ['c','c++','python','java','kotlin']
y = [10,20,30,40,50]
z = [5,15,25,35,45]

# BAR PLOT
plt.bar(x,y,color='y',edgecolor='r',width=0.5,label='Graph 1')
plt.bar(x,z,color='b',edgecolor='r',width=0.5,label='Graph 2')
plt.legend()
plt.show()

# SCATTER PLOT
plt.scatter(x,y,c='y',edgecolor='r',marker='*')
plt.show()

# HIST PLOT
plt.hist(y,color='b',edgecolor='r',bins=20,orientation='horizontal')
plt.show()
plt.hist(y,color='b',edgecolor='r',histtype='step')
plt.show()

# PIE CHART
ex = [0.4,0,0,0,0]
plt.pie(y,labels=x,explode=ex,autopct="%0.01f%%")
plt.show()

# STACK PLOT
a1 = [2,3,2,5,4]
a2 = [2,3,4,5,6]
a3 = [1,3,2,4,2]
l = ['area 1','area 2','area 3']
plt.stackplot(y,a1)
plt.show()
plt.stackplot(y,a1,a2,a3,labels=l)
plt.legend()
plt.show()

# BOX PLOT
plt.boxplot(y,widths=0.3,label='Python',patch_artist=True,showmeans=True,)
plt.show()
plt.boxplot(y,vert=False,widths=0.3)
plt.show()
lists = [y,z]
plt.boxplot(lists,labels=['Python','c++'],showmeans=True)
plt.show()

# STEP PLOT
plt.step(y,z,color = 'r',marker='o')
plt.grid()
plt.show()

# LINE GRAPH
plt.plot(y,z)
plt.text(30,10,'Hello There')
plt.annotate('Python',xy=(20,20),xytext=(30,30),arrowprops=dict(facecolor='black'))
plt.show()

# STEM DIAGRAM
plt.stem(y,z,linefmt=':',markerfmt='r*')
plt.show()

# FILL BETWEEN
plt.plot(y,z,color= 'red')
plt.fill_between(y,z)
plt.show()

plt.plot(y,z,color= 'red')
plt.fill_between(x=[20,40],y1 = 20, y2 = 40)
plt.show()

n1 = np.array(y)
n2 = np.array(z)

plt.plot(y,z,color= 'red')
plt.fill_between(n1,n2,color = 'g', where=(n1 >= 20) & (n1 <= 40))
plt.show()

plt.bar(x,y,width=0.3,color=c,edgecolor="m",linewidth=4,label="usage")

plt.bar(x,y,width=0.3,color="b",edgecolor="m",linewidth=4,label="usage1")
plt.bar(x,z,width=0.3,color="r",edgecolor="pink",linewidth=4,label="usage2")

plt.scatter(x,y,c=color,s=z,marker="*",edgecolor="black",linewidth=2)

plt.hist(z,color="r",edgecolor="black",linewidth=2)
plt.hist(z,"auto",(0,100),edgecolor="black",linewidth=2)#auto= suitable no of bars for the range given
plt.hist(z,color="r",bins=6,edgecolor="black",linewidth=2,cumulative=-4)#bins=no of bars
plt.hist(z,color="r",bins=6,edgecolor="black",linewidth=2,histtype="step")#type step=no filling clr only border
plt.hist(z,color="r",bins=6,edgecolor="black",linewidth=2,orientation="horizontal")#bins=no of bars

plt.pie(y, labels=x, colors=color, explode=z, autopct="%d")  # autopct = labeled values inside the pie chart

plt.stackplot(y,a1,a2,a3,labels=x)#can pass 1 or more a's

plt.boxplot(y,vert=False,widths=0.5) #for horizontal
plt.boxplot(y,labels=["Hi"],patch_artist=True,showmeans=True)#patch artist color, show mean= line and green arrow rep mean
plt.boxplot(x,labels=["Hi","Bye"],showmeans=True)

plt.step(y,z,color="b",marker="*")

plt.plot(z,y)
plt.text(20,30,"Hi",fontsize=12)#adding text to a perticular point
plt.annotate("Bye",xy=(40,30),xytext=(60,10),arrowprops=dict(facecolor="green"))#pointing using arrow xy text to xy

plt.stem(y, z, linefmt=":", markerfmt="*")

plt.subplot(2, 2, 1)
plt.plot(y, z, color="r")
plt.subplot(2, 2, 2)
plt.pie(y)
plt.subplot(2, 2, 3)
plt.bar(y, z)

plt.fill_between(y, z)
plt.fill_between([17, 20], 11, 21)
plt.fill_between(y, z, color="b", where=(y > 20) & (z < 20), label="Conditional filling")
plt.tight_layout()
# ------------------------------------------------------PANDAS--------------------------------------------------------
import pandas as pd
import numpy as np
import random

# Initial data
data = {'Name': ["Adina", "Rafia", "Sara"], 'age': [12, 11, 2]}
mydata = [1111, 2222, 3333, 4444]
myIndex = ["usa", "canada", "mexico", "germany"]
mydata = pd.Series(data=mydata, index=myIndex)
print(mydata)
print(mydata.iloc[0])
print(mydata["usa"])

data = pd.DataFrame(data)
print(data)
print(data.shape)

# Reading CSV file
data = pd.read_csv(r"heart.csv")
print(data)
print(data.head())
print(data.tail())

# Changing the data type of 'sex' column to string before replacement
data['sex'] = data['sex'].astype(str)

# Replacing 0 with 'female' and 1 with 'male'
data['sex'] = data['sex'].replace({'0': 'female', '1': 'male'})

# Renaming columns
data = data.rename(columns={'rest_ecg': 'ecg', 'sex': 'gender'})
print(data.describe())

# List of names
names = ['Ali', 'Salman', 'Sohail', 'Mohsin', 'Waqas', 'Zeshan', 'Babar', 'John', 'Musk', 'Elon', 'Michael']

# Check if the number of names matches the number of rows
#if len(names) != len(data):
 #   raise ValueError(f"Number of names ({len(names)}) does not match number of rows in data ({len(data)}).")

# Insert names into the dataframe
#data.insert(0, "name", names)
#print(data)

#make a new dataframe of 5 rows
newData = data.head()

#then select specific columns
newData= data[['age', 'gender', 'cp', 'target']]
print(newData)

#inplace = True changes in the existing dataframe while inpllace = False creats a copy
#dropping column
data.drop(labels=['target'], axis = 1, inplace = False)
print(data)

data= pd.read_csv(r'students.csv')
print(data)

data.drop(labels=['Grade'], axis = 1, inplace = True)
print(data)

#Adding a new row
new_row = {"Student_ID":"1001", "Name": "Sara", "Age":13, "City":"Karachi"}
new_row_df = pd.DataFrame([new_row])
data = pd.concat([data, new_row_df], ignore_index = True)

print(data)

#inserting a new row at specific location
data.loc[11] = [1002, "Adina", 19, "Hyderabad"]
print(data)

#dropping rows
newdf = data.drop(labels= [10, 11], axis = 'rows')
print(newdf)

#dropping columns
newdf = data.drop("City", axis = 'columns')
print(newdf)

print(data)


#The ignore_index parameter in pandas is used 
#to reset the index when concatenating or 
#appending data to a DataFrame. By default, 
#pandas tries to preserve the original index 
##Setting ignore_index=True ensures that the 
#resulting DataFrame has a sequentially updated 
#index, starting from 0,
## without keeping the original row indices.'''

#combining data frames
data2 = pd.read_csv(r'new_students.csv')
print(data2)

result = pd.concat([data, data2], axis = 0, ignore_index = False)
print(result)

print(data)

data = data.set_index('Name')
print(data)
data = data.reset_index()
print(data)

print(data.loc['Ali':'Babar']['Age'])
print(data.loc[data['Age']>15])

data.sort_values('Age', inplace=True)
print(data)

#changing data at a particular position
data.at[10, 'Age'] = 19.0
print(data)

#making derived column
#data['new_col'] = data['City']+data['Student_ID']
print(data)

data['attendance'] = np.nan
print(data)
data = data.dropna(axis =1 , how = 'all')
print(data)

data['Student_ID'] = data['Student_ID'].astype('int64')
print(data)

#getting first 2 characters of Name
print(data['Name'].str[:2])

data.to_csv(r'New_file.csv', index = True, header = False)

registrations = pd.DataFrame({'reg_id': [1,2,3,4], 'name': ['Adina', 'Sara', 'Hafsa', 'Arwa']})
login = pd.DataFrame({'login_id': [1,2,3,4], 'name':['Adina', 'Raghib', 'Rahman', 'Hafsa']})
print(registrations)
print(login)

print(pd.merge(registrations, login, how = 'inner', on = 'name'))
print(pd.merge(registrations, login, how = 'outer', on = 'name'))

#groupby
#print(df.groupby('model_year).mean())
#group by multiple columns
#print(df.groupby(['model_year', 'cylinders']).mean())

#between--->returns bool values
print(login['login_id'].between(2,3,inclusive = 'both'))

print(data.nlargest(5, 'Age'))
print(data.nsmallest(5, 'Age'))

#.apply()
def my_func(x, h, l):
    if x > h:
        return("high")
    if x > l:
        return("medium")
    else:
        return("low")
    
print(login["login_id"].apply(my_func, args = [10, 3]))

def mode(x):
    return x.mode()

print(data.apply(mode, axis = 0))

def yelp(price):
    if price > 100:
        return '$'
    if price >= 10 and price <=100:
        return '$$'
    else:
        return '$$$'
    
data['Elder']= data['Age'].apply(yelp)
print(data)
