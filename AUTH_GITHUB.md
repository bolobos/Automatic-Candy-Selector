# 🔐 Authentification GitHub - Solutions

## ✅ Option 1: Personal Access Token (PAT) - Recommandé

### Créer un token:
1. Aller sur: https://github.com/settings/tokens
2. Cliquer **"Generate new token"** → **"Generate new token (classic)"**
3. Cocher: `repo` (accès complet aux dépôts)
4. Cliquer **"Generate token"**
5. **COPIER LE TOKEN** (il ne sera plus visible après)

### Utiliser le token:
```powershell
# Windows - Enregistrer dans le Credential Manager
git config --global credential.helper wincred

# Lors du prochain push, entrer:
# Username: bolobos
# Password: <VOTRE_TOKEN>
```

Le token sera sauvegardé automatiquement.

---

## 🔑 Option 2: SSH (Plus sécurisé, pas de mot de passe)

### Générer une clé SSH:
```powershell
wsl bash -c "ssh-keygen -t ed25519 -C 'votre.email@example.com'"
# Appuyer sur Entrée 3 fois (pas de passphrase)
```

### Afficher la clé publique:
```powershell
wsl bash -c "cat ~/.ssh/id_ed25519.pub"
```

### Ajouter à GitHub:
1. Aller sur: https://github.com/settings/keys
2. Cliquer **"New SSH key"**
3. Coller la clé publique
4. Cliquer **"Add SSH key"**

### Changer l'URL du remote:
```powershell
git remote set-url origin git@github.com:bolobos/Automatic-Candy-Selector.git
```

---

## 🌐 Option 3: GitHub CLI (Plus simple)

```powershell
# Installer GitHub CLI
winget install GitHub.cli

# Authentification
gh auth login
# Choisir: GitHub.com → HTTPS → Login with a web browser
```

---

## 🚀 Pousser après authentification

```powershell
git push origin trompe-d-eustache-yolo
```
