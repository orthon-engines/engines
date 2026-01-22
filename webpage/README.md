# Ørthon Documentation Site

A Jekyll-based documentation site using the Editorial theme style.

## Quick Start (Local Development)

### 1. Install Ruby and Jekyll

**Mac:**
```bash
brew install ruby
gem install bundler jekyll
```

**Windows:**
Download Ruby+Devkit from https://rubyinstaller.org/

**Linux:**
```bash
sudo apt install ruby-full build-essential
gem install bundler jekyll
```

### 2. Install Dependencies

```bash
cd orthon-docs
bundle install
```

### 3. Run Locally

```bash
bundle exec jekyll serve
```

Visit `http://localhost:4000` in your browser.

### 4. Make Changes

- Edit pages in `_docs/` folder (Markdown files)
- Edit styles in `assets/css/main.css`
- Edit navigation in `_layouts/default.html`

Changes auto-reload when you save.

---

## Deploy to GitHub Pages

1. Create a new repo on GitHub (e.g., `orthon-docs`)

2. Push this folder:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOURUSERNAME/orthon-docs.git
git push -u origin main
```

3. Go to repo **Settings → Pages → Source** → select `main` branch

4. Your site will be live at `https://YOURUSERNAME.github.io/orthon-docs/`

---

## File Structure

```
orthon-docs/
├── _config.yml          # Site settings
├── _layouts/
│   ├── default.html     # Main template (sidebar + content)
│   └── doc.html         # Documentation page template
├── _docs/               # Your documentation (Markdown)
│   ├── overview.md
│   ├── vector-layer.md
│   └── quickstart.md
├── assets/
│   ├── css/main.css     # Styling
│   └── js/main.js       # Sidebar toggle
├── index.html           # Home page
├── Gemfile              # Ruby dependencies
└── README.md            # This file
```

## Adding New Pages

1. Create a new `.md` file in `_docs/`:

```markdown
---
title: My New Page
description: A brief description
---

Your content here...
```

2. Add a link in `_layouts/default.html` sidebar menu

3. Refresh your browser

---

## Credits

- Theme style based on [Editorial by HTML5 UP](https://html5up.net/editorial)
- Built with [Jekyll](https://jekyllrb.com/)
