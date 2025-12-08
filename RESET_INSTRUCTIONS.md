# 🗑️ Complete Repository Reset

To delete and recreate your repository cleanly:

## Step 1: Delete Repository on GitHub

1. Go to https://github.com/arrdel/mdlm-atat
2. Click **Settings** (gear icon)
3. Scroll to bottom → **Delete this repository**
4. Type the repository name to confirm
5. Click **I understand the consequences, delete this repository**

## Step 2: Wait 30 seconds for deletion to complete

## Step 3: Create New Repository

1. Go to https://github.com/new
2. Repository name: `mdlm-atat`
3. Description: "Masked Discrete Latent Models with Adaptive Token-level Training"
4. Visibility: Public or Private
5. **Do NOT** check "Initialize with README"
6. Click **Create repository**

## Step 4: Push Clean Code

```bash
cd /home/adelechinda/home/projects/mdlm

# Configure new remote
git remote remove origin
git remote add origin https://github.com/arrdel/mdlm-atat.git

# Push
git push -u origin master
```

That's it! Brand new, clean repository with only your work.
