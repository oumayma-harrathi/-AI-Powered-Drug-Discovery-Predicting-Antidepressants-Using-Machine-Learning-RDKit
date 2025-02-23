import pandas as pd
from rdkit import chem
from rdkit.chemc import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def extract_features(smiles):
    #fonction prend un SMILES (représentation textuelle d'une molécule) 
    #et retourne un dictionnaire contenant ses caractéristiques chimiques
    
    mol=chem.MolFromSmiles(smiles)
    if mol is None :
        return None
    
    
    features={
         "MolWt": Descriptors.MolWt(mol),  # Poids moléculaire
        "LogP": Descriptors.MolLogP(mol),  # Lipophilie (solubilité dans les graisses)
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),  # Nombre d'accepteurs de liaisons H
        "NumHDonors": Descriptors.NumHDonors(mol),  # Nombre de donneurs de liaisons H
        "RingCount": Descriptors.RingCount(mol)  # Nombre d'anneaux (cycles) dans la molécule
    }
    
    return features

data={
    "Smiles": [
        COc1ccc2[nH]c(=O)cc2c1",  # Mirtazapine
        "CC(=O)Oc1ccccc1C(=O)O",  # Aspirine (Non antidépresseur)
        "CN(C)C(=O)c1ccccc1O",  # Paroxetine
        "CN(C)C(=O)c1ccc(Cl)cc1",  # Autre molécule
    ],
    "Label": [1, 0, 1, 0]  # 1 = Antidépresseur, 0 = Non antidépresseur
}
df=pd.DataFrame(data)


features_list = []
labels = []

for i, row in df.iterrows():
    features = extract_features(row["SMILES"])  
    if features:  #mol valide 
        features_list.append(list(features.values()))
        labels.append(row["Label"])
        
        
        
X = pd.DataFrame(features_list, columns=["MolWt", "LogP", "NumHAcceptors", "NumHDonors", "RingCount"])
y = labels  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  

y_pred = model.predict(X_test)
print("Précision du modèle:", accuracy_score(y_test, y_pred))  

new_molecule = "CN(C)C(=O)c1ccccc1"  
features_test = extract_features(new_molecule)  

if features_test:
    prediction = model.predict([list(features_test.values())])  
    print("Cette molécule est-elle un antidépresseur ?", "Oui" if prediction[0] == 1 else "Non")
else:
    print("Erreur : Molécule invalide")



        
    