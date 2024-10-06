Here’s a **30-minute crash course on LaTeX programming** to help you get started with creating documents containing **complex mathematical equations, images, and formatting** for your data science book.

### Step-by-Step Crash Course Outline:

#### 1. **Set Up LaTeX Environment** (2 mins)
You can either:
- **Install LaTeX locally**: Use TeX distributions like [TeX Live](https://www.tug.org/texlive/) or [MiKTeX](https://miktex.org/).
- **Use Online LaTeX Editors**: 
  - [Overleaf](https://www.overleaf.com/) is the most popular online LaTeX editor, ideal for beginners.

#### 2. **Basic Document Structure** (3 mins)
Every LaTeX document has a consistent structure:
```latex
\documentclass{article}   % Document class (article, report, book, etc.)
\usepackage{amsmath}      % For advanced mathematical equations
\usepackage{graphicx}     % For inserting images
\usepackage{hyperref}     % For hyperlinks (optional)

\title{Your Title}
\author{Your Name}

\begin{document}
\maketitle               % Title generation
\tableofcontents         % Generates the table of contents automatically
\section{Introduction}
Your content goes here...
\end{document}
```
**Key Commands**:
- `\documentclass{}`: Defines the document type. Common types are `article`, `book`, `report`.
- `\usepackage{}`: Includes necessary packages (e.g., `amsmath` for math, `graphicx` for images).
- `\maketitle`: Generates the document title.
- `\tableofcontents`: Automatically generates a table of contents from the section titles.

#### 3. **Mathematical Equations** (10 mins)
LaTeX shines with its ability to format complex mathematical equations. You can write equations in:
- **Inline Mode**: Enclosed within `$ $`.
- **Display Mode**: Enclosed within `\[ \]` or `\begin{equation} \end{equation}`.

**Common Mathematical Symbols**:
```latex
% Inline Math Example
This is an inline equation: $y = mx + b$

% Display Math Example
\[
    E = mc^2
\]

% Equation Numbering
\begin{equation}
    f(x) = \int_{-\infty}^{\infty} e^{-x^2} dx
\end{equation}
```
**Common Symbols**:
- Superscript: `x^2`
- Subscript: `x_i`
- Fractions: `\frac{a}{b}`
- Square root: `\sqrt{x}`
- Summation: `\sum_{i=1}^{n} x_i`
- Integral: `\int_{a}^{b} f(x) dx`

**Matrices** (often used in data science):
```latex
\[
    A = \begin{bmatrix}
    a_{11} & a_{12} \\
    a_{21} & a_{22}
    \end{bmatrix}
\]
```
For more complex formatting, use the `amsmath` package to extend functionality (already included in the setup).

#### 4. **Inserting Images** (5 mins)
To include images in your document, use the `graphicx` package. Here’s how to insert and manipulate images:

1. **Place the image file** (e.g., `image.png`) in the same folder as your `.tex` file.
2. **Insert image in the document**:
```latex
\begin{figure}[h]
    \centering
    \includegraphics[width=0.6\textwidth]{image.png}
    \caption{This is an example image.}
    \label{fig:example_image}
\end{figure}
```
- `\includegraphics[width=0.6\textwidth]{image.png}`: Includes the image and scales it to 60% of the text width.
- `\caption{}`: Adds a caption under the image.
- `\label{}`: Labels the image for referencing later (e.g., `Figure \ref{fig:example_image}`).

#### 5. **Tables and Figures for Data Science** (5 mins)
You’ll frequently need to create tables, especially for presenting results in data science.

**Creating Tables**:
```latex
\begin{table}[h]
\centering
\begin{tabular}{|c|c|c|}
\hline
Feature & Importance & Rank \\
\hline
Feature 1 & 0.75 & 1 \\
Feature 2 & 0.58 & 2 \\
\hline
\end{tabular}
\caption{Feature importance ranking}
\label{tab:feature_importance}
\end{table}
```
- `|c|c|c|`: Defines 3 columns with vertical lines between them.
- `\hline`: Horizontal line.
- `\caption{}`: Adds a caption to the table.

#### 6. **Cross-referencing** (3 mins)
LaTeX allows you to cross-reference equations, figures, and tables using `\label{}` and `\ref{}`.

For example, if you label an equation as `\label{eq:sample_eq}`, you can refer to it in the text as `Equation \ref{eq:sample_eq}`.

**Example**:
```latex
As shown in Figure \ref{fig:example_image}, the model converges quickly.
```

#### 7. **Customizing Sections and Subsections** (2 mins)
LaTeX allows you to structure your document into sections, subsections, and subsubsections:

```latex
\section{Introduction}
This is a section.

\subsection{Background}
This is a subsection.

\subsubsection{Detailed Explanation}
This is a subsubsection.
```
Each of these sections is automatically numbered and added to the table of contents (if `\tableofcontents` is included).

#### 8. **Lists** (2 mins)
You can create ordered and unordered lists easily.

**Unordered List**:
```latex
\begin{itemize}
    \item First item
    \item Second item
    \item Third item
\end{itemize}
```

**Ordered List**:
```latex
\begin{enumerate}
    \item Step one
    \item Step two
    \item Step three
\end{enumerate}
```

#### 9. **Citations and Bibliography** (3 mins)
For academic writing or a book, you’ll need to include citations and a bibliography.

1. **Include the `biblatex` package**:
```latex
\usepackage{biblatex}
\addbibresource{references.bib}  % BibTeX file with references
```

2. **Add citations in the document**:
```latex
According to \cite{author2021}, deep learning has achieved great results.
```

3. **Bibliography section**:
```latex
\printbibliography
```

4. **Example of a `.bib` file (`references.bib`)**:
```bibtex
@article{author2021,
  title={Title of the Paper},
  author={Author, A.},
  journal={Journal Name},
  year={2021},
  volume={15},
  pages={123-456}
}
```

#### 10. **Practice and Testing** (5 mins)
Now that you’ve learned the basics, practice creating a sample LaTeX document that includes:
- A title, sections, and subsections.
- Mathematical equations (both inline and display).
- A table and an image.
- Cross-referencing figures, tables, and equations.
  
Use Overleaf or your local LaTeX setup to compile the document and make adjustments.

---

### Quick Reference Cheat Sheet:

- **Sections**: `\section{}`, `\subsection{}`, `\subsubsection{}`
- **Mathematical Equations**:
  - Inline: `$ y = mx + b $`
  - Display: `\begin{equation} ... \end{equation}`
  - Matrices: `\begin{bmatrix} ... \end{bmatrix}`
- **Images**:
  ```latex
  \includegraphics[width=0.6\textwidth]{image.png}
  \caption{}
  \label{}
  ```
- **Tables**:
  ```latex
  \begin{tabular}{|c|c|c|}
  \hline
  ...
  \end{tabular}
  ```
- **Lists**:
  - Unordered: `\begin{itemize} \item ... \end{itemize}`
  - Ordered: `\begin{enumerate} \item ... \end{enumerate}`

---

This crash course should get you up and running with LaTeX quickly, allowing you to write documents with complex mathematical equations, tables, and images for your data science book!