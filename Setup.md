# Setup

## Setup Local Notebook Workspace

1. Install Juypter, Notebook, Volia as per Juypter.org
2. Create, Activate and Se;ect Virtual Environment
3. Install `ipukernel`
4. Create new kernel
5. Start Juypter
6. Select Kernel for project
   6.1 Create/Import a (New) Juypter Notebook
   6.2 Chooe your Notebook's Kernel

> Sources: [1] [Video: How To Setup & Run Jupyter Notebooks in VS Code w/ Virtual Environment & Kernels (Remote & Local) - Devin Schumacher](https://youtu.be/-j6y-5t37os)  
> Sources: [2] [Article: How To Setup & Run Jupyter Notebooks in VS Code w/ Virtual Environment & Kernels (Remote & Local) - devinschumacher.com](https://devinschumacher.com/how-to-setup-jupyter-notebook-virtual-environment-vs-code-kernels/)

### Installation Flow

#### 1. Install Juypter, Notebook, Volia

- Use `pip install` as not using conda or mamba package management, on Windows, VSCode
- Use Brew for WSL, Linux, macOS to install
- Use `venv` per project environments over _user_ enviornments (`--user`) and _system_ level environments for packages.

#### JuypterLab

```python
pip install jupyterlab

# Launch 

juypter lab

```

<details>
<summary>Install Log</summary>
Installing collected packages:
<pre>
 anyio, 
 argon2-cffi-bindings, 
 argon2-cffi, 
 arrow, 
 async-lru, 
 attrs, 
 babel, 
 beautifulsoup4, 
 bleach, 
 certifi,
 cffi, 
 charset-normalizer, 
 defusedxml, 
 fastjsonschema, 
 fqdn, 
 h11, 
 httpcore, 
 httpx, 
 idna, 
 isoduration, 
 jinja2, 
 json5, 
 jsonpointer, 
 jsonschema-specifications, 
 jsonschema, 
 jupyter-events, 
 jupyter-lsp, 
 jupyter-server-terminals, 
 jupyter-server, 
 jupyterlab
 jupyterlab-pygments, 
 jupyterlab-server, 
 MarkupSafe, 
 mistune, 
 nbclient, 
 nbconvert, 
 nbformat, 
 notebook-shim, 
 overrides, 
 pandocfilters, 
 prometheus-client, 
 pycparser, 
 python-json-logger, 
 pyyaml, pywinpty, 
 referencing, 
 requests, 
 rfc3339-validator, 
 rfc3986-validator, 
 rpds-py, 
 send2trash, 
 setuptools, 
 sniffio, 
 soupsieve, 
 terminado, 
 tinycss2, 
 types-python-dateutil, 
 uri-template, 
 urllib3, 
 webcolors, 
 webencodings, 
 websocket-client, 
</pre>
</details>

<details>
<summary>Run Log</summary>
<pre>
[I 2024-07-05 17:53:13.089 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2024-07-05 17:53:13.099 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2024-07-05 17:53:13.110 ServerApp] jupyterlab | extension was successfully linked.
[I 2024-07-05 17:53:13.120 ServerApp] notebook | extension was successfully linked.
[I 2024-07-05 17:53:13.128 ServerApp] Writing Jupyter server cookie secret to C:\Users\Charles\AppData\Roaming\jupyter\runtime\jupyter_cookie_secret
D:\Code\IBMSB\.venv\Lib\site-packages\traitlets\traitlets.py:1897: DeprecationWarning: ServerApp.token config is deprecated in jupyter-server 2.0. Use IdentityProvider.token
  return t.cast(Sentinel, self._get_trait_default_generator(names[0])(self))
[I 2024-07-05 17:53:13.659 ServerApp] notebook_shim | extension was successfully linked.
[I 2024-07-05 17:53:13.660 ServerApp] voila.server_extension | extension was successfully linked.
[I 2024-07-05 17:53:13.719 ServerApp] notebook_shim | extension was successfully loaded.
[I 2024-07-05 17:53:13.722 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2024-07-05 17:53:13.722 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2024-07-05 17:53:13.728 LabApp] JupyterLab extension loaded from D:\Code\IBMSB\.venv\Lib\site-packages\jupyterlab
[I 2024-07-05 17:53:13.728 LabApp] JupyterLab application directory is D:\Code\IBMSB\.venv\share\jupyter\lab      
[I 2024-07-05 17:53:13.729 LabApp] Extension Manager is 'pypi'.
[I 2024-07-05 17:53:14.035 ServerApp] jupyterlab | extension was successfully loaded.
[I 2024-07-05 17:53:14.044 ServerApp] notebook | extension was successfully loaded.
[I 2024-07-05 17:53:14.059 ServerApp] voila.server_extension | extension was successfully loaded.
[I 2024-07-05 17:53:14.060 ServerApp] Serving notebooks from local directory: D:\Code\IBMSB
[I 2024-07-05 17:53:14.060 ServerApp] Jupyter Server 2.14.1 is running at:
[I 2024-07-05 17:53:14.060 ServerApp] http://localhost:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41
[I 2024-07-05 17:53:14.061 ServerApp]     http://127.0.0.1:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41
[I 2024-07-05 17:53:14.061 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2024-07-05 17:53:14.147 ServerApp]

    To access the server, open this file in a browser:
        file:///C:/Users/Charles/AppData/Roaming/jupyter/runtime/jpserver-37228-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41
        http://127.0.0.1:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41

[I 2024-07-05 17:53:14.368 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server
[W 2024-07-05 17:53:36.146 LabApp] Could not determine jupyterlab build status without nodejs

</pre>
</details>

1. To access the server, open this file in a browser:
   - file:///C:/Users/Charles/AppData/Roaming/jupyter/runtime/jpserver-37228-open.html
2. Or copy and paste one of these URLs:
   - http://localhost:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41
   - http://127.0.0.1:8888/lab?token=7ebb81a71da0b99bda657c46dea3a8903d24528a0de6cf41
3. Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

#### Juypter Notebook

```python
pip install notebook

# Run Notebook

jupyter notebook

```

<details>
<summary>Install Log</summary>
Requirement already satisfied as per jupyterlab
Installing collected packages:
<pre>
 notebook
</pre>
</details>

#### Voilà - CLI | Web Application Server

```python
pip install voila

```

### VSCode Juytper Extensions

`3rd Party Bill of Materials - Extensions`

> - ✅ Jupyter (extension ID: ms-toolsai.jupyter)
> - ✅ Jupyter Notebook Renderers (extension ID: ms-toolsai.jupyter-renderers)
> - ✅ Jupyter Keymap (extension ID: ms-toolsai.jupyter-keymap)
> - ✅ Jupyter Cell Tags (extension ID: ms-toolsai.vscode-jupyter-cell-tags)
> - ✅ Jupyter Slide Show (extension ID: ms-toolsai.vscode-jupyter-slideshow)
> - ✅ VS Code Jupyter Notebook Previewer (extension ID: jithurjacob.nbpreviewer)
> - ✅ Juyptext for Notebooks (conguiwu) (extension ID: congyiwu.vscode-jupytext)
> - ✅ Default Python Kernels for Juypter Notebooks (extension ID: donjayamanne.vscode-default-python-kernel)
> - ✅ Juypter TOC (extension ID: xelad0m.jupyter-toc)

**Other | Related**

> - ✅ GitHub Issue Notebooks (extension ID: ms-vscode.vscode-github-issue-notebooks)
> - ✅ Polyglot Notebooks (extension ID: ms-dotnettools.dotnet-interactive-vscode )

---

### Project Setup Flow

#### 2. Create, activate & select the virtual environment

- Either: i. Clone your repo from GitHub, navigate to and start enabling Python per project
- Or : ii. Make a new repo directory, initialise Git for project and start enabling Python per project

##### Create `venv`

```python
# syntax
python3 -m venv <virtual environment name>

# example
# that would create a virtual environment named 'myenv'
python3 -m venv myenv

```

```python
# Check for Pip Version@Latest
# > D:\Code\IBMSB via 🐍 v3.12.4 (.venv) 

python -m pip install --upgrade pip

# Requirement already satisfied: pip in d:\langs\python\312\lib\site-packages (24.0)
# Successfully installed pip-24.1.1

```

##### Activate

```python
# syntax
# source <virtual environment name>/bin/activate

# example (Windows/pwsh)
.venv\\Scripts\\activate.ps1

# VS CODE prompts you to set it as default for the project, hit yes.

```

### Kernel Setup Flow

> Sources: [1,2]

#### 3. Install `ipykernel`

- Provides the Python kernel that allows Jupyter to execute Python code
- The IPython kernel, provided by ipykernel, is the default kernel used for executing Python code in Jupyter.
- Kernels uust match Python environment, so hence using a name `venv` environement is critical for correct package dependencies.

```python
pip3 install ipykernel

```

<details>
<summary> Install log</summary>
<pre>Installing collected packages: 
wcwidth, 
pywin32, 
pure-eval, 
traitlets, 
tornado, 
six, 
pyzmq, 
pygments, 
psutil, 
prompt-toolkit, 
platformdirs, 
parso, 
packaging, 
nest-asyncio, 
executing, 
decorator, 
debugpy, 
colorama, 
python-dateutil, 
matplotlib-inline, 
jupyter-core, 
jedi, 
comm, 
asttokens, 
stack-data, 
jupyter-client, 
ipython, 
ipykernel#
</pre>

<pre>
Successfully installed 
asttokens-2.4.1 
colorama-0.4.6 
comm-0.2.2 
debugpy-1.8.2 
decorator-5.1.1 
executing-2.0.1 
ipykernel-6.29.5 
ipython-8.26.0 
jedi-0.19.1 
jupyter-client-8.6.2 
jupyter-core-5.7.2 
matplotlib-inline-0.1.7 
nest-asyncio-1.6.0 
packaging-24.1 
parso-0.8.4 
platformdirs-4.2.2 
prompt-toolkit-3.0.47 
psutil-6.0.0 pure-eval-0.2.2 
pygments-2.18.0 
python-dateutil-2.9.0.post0 
pywin32-306 pyzmq-26.0.3 
six-1.16.0 stack-data-0.6.3 
tornado-6.4.1 
traitlets-5.14.3 
wcwidth-0.2.13</pre>
</details>

#### 4. Create New (Named) Kernel

> Sources: [1,2]

```python
# syntax
# python3 -m ipykernel install --user --name=<projectname>

# example: Create a kernel named 'myproject'
python3 -m ipykernel install --user --name=myproject

```

<details>
<summary>Terminal LogL </code>ibmsbai</code> </summary>
D:\Code\IBMSB via 🐍 v3.12.4 (.venv) 
<pre>
python -m ipykernel install --user --name=ibmsbai 
Installed kernelspec ibmsbai in C:\Users\Charles\AppData\Roaming\jupyter\kernels\ibmsbai
</pre>
</details>

#### 5. Start Juypter

> Sources: [1,2]

- From Terminal: as follows
- From Command Panel:
  - ![alt text](assets\img_Commands_Juypter.png)

```python
jupyter notebook

```

- Accessing Notebook
  - Localhost or 127.0.0.1
  - Port: 8889

```python
# Expected Output (Update for current machine/environment)

To access the notebook, open this file in a browser:
        file:///Users/<your username>/Library/Jupyter/runtime/nbserver-15044-open.html
    Or copy and paste one of these URLs:
        https://localhost:8889/?token=f1ae910e56381c26a62cfb18f83241076bd11d84f7e8e36e
     or https://127.0.0.1:8889/?token=f1ae910e56381c26a62cfb18f83241076bd11d84f7e8e36e

```

#### 6. Select Kernet for Project

> Sources: [1,2]

##### 6.1 Create new notebook: `.iynb`

![alt text](assets\img_Command_Notebooks1.png)<br>
![alt text](assets\img_Command_Notebooks2.png)

##### 6.2 Choose your kernel

![alt text](assets\img_Command_Kernels.png)

---

### Language Servers Flow

#### 7. Python Language Servers

#### 7.1 Jedi LSP

```python
pip install jedi-language-server

```

##### 7.2 Python LSP Server | Protocol

```python
pip install python-lsp-server

```

#### 7.3 Pylance

```python
pip install pylance

```

<details>
<summary>Install Log</summary>
Installing collected packages: 
<pre>
numpy, 
pyarrow, 
pylance
</pre>
</details>

#### 8. NPM / Node Language Servers

---

### Juypter Lab

- Launch with `jupyter lab`
- Create a PWA in Edge, pin to star and task bar.
- Check the generated URL in terminal

> URI: [Jupyter Lab | LocalHost:8888](http://localhost:8888/lab)

![alt text](assets\img_Running_JupyterLab_Local.png)

### Launch an Interactive Window

- `Ctrl + shift + P` + `Create Interactive Window`
  - Multiple Workspace folders opened d:\Code\IBMSB, d:\Langs\Node\v20.15.0
    - Starting interactive window for resource '' with controller `'.jvsc74a57bd089d947ee5d2aaa1eefb549a877eacfb210543ead6485de83f2532087dfd2eefb.d:\Code\IBMSB\.venv\Scripts\python.exe.d:\Code\IBMSB\.venv\Scripts\python.exe.-m#ipykernel_launcher (Interactive)'`

### Start Jupyter Server

- `juypter select intrepter to start Jupyter server `0
  - Process Execution: d:\Code\IBMSB\.venv\Scripts\python.exe -c "import pip;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
  - Process Execution: d:\Code\IBMSB\.venv\Scripts\python.exe -c "import jupyter;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
  - Process Execution: d:\Code\IBMSB\.venv\Scripts\python.exe -c "import notebook;print('6af208d0-cb9c-427f-b937-ff563e17efdf')"
  - Process Execution: d:\Code\IBMSB\.venv\Scripts\python.exe -m jupyter kernelspec --version