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

#### Jupyter

```python
pip install jupyter

```

<details>
<summary>Juypter</summary>
pip install jupyter    
Collecting jupyter
  Downloading jupyter-1.0.0-py2.py3-none-any.whl.metadata (995 bytes)
Requirement already satisfied: notebook in d:\code\ibmsb\.venv\lib\site-packages (from jupyter) (7.2.1)
Collecting qtconsole (from jupyter)
  Downloading qtconsole-5.5.2-py3-none-any.whl.metadata (5.1 kB)
Collecting jupyter-console (from jupyter)
  Downloading jupyter_console-6.6.3-py3-none-any.whl.metadata (5.8 kB)
Requirement already satisfied: nbconvert in d:\code\ibmsb\.venv\lib\site-packages (from jupyter) (7.16.4)
Requirement already satisfied: ipykernel in d:\code\ibmsb\.venv\lib\site-packages (from jupyter) (6.29.5)
Collecting ipywidgets (from jupyter)
  Downloading ipywidgets-8.1.3-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: comm>=0.1.1 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (0.2.2)
Requirement already satisfied: debugpy>=1.6.5 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (1.8.2)
Requirement already satisfied: ipython>=7.23.1 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (8.26.0)
Requirement already satisfied: jupyter-client>=6.1.12 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (8.6.2)
Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (5.7.2)
Requirement already satisfied: matplotlib-inline>=0.1 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (0.1.7)
Requirement already satisfied: nest-asyncio in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (1.6.0)
Requirement already satisfied: packaging in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (24.1)
Requirement already satisfied: psutil in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (6.0.0)
Requirement already satisfied: pyzmq>=24 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (26.0.3)
Requirement already satisfied: tornado>=6.1 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (6.4.1)
Requirement already satisfied: traitlets>=5.4.0 in d:\code\ibmsb\.venv\lib\site-packages (from ipykernel->jupyter) (5.14.3)
Collecting widgetsnbextension~=4.0.11 (from ipywidgets->jupyter)
  Downloading widgetsnbextension-4.0.11-py3-none-any.whl.metadata (1.6 kB)
Collecting jupyterlab-widgets~=3.0.11 (from ipywidgets->jupyter)
  Downloading jupyterlab_widgets-3.0.11-py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: prompt-toolkit>=3.0.30 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-console->jupyter) (3.0.47)
Requirement already satisfied: pygments in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-console->jupyter) (2.18.0)
Requirement already satisfied: beautifulsoup4 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (4.12.3)
Requirement already satisfied: bleach!=5.0.0 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (6.1.0)
Requirement already satisfied: defusedxml in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (0.7.1)
Requirement already satisfied: jinja2>=3.0 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (3.1.4)
Requirement already satisfied: jupyterlab-pygments in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (0.3.0)
Requirement already satisfied: markupsafe>=2.0 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (2.1.5)
Requirement already satisfied: mistune<4,>=2.0.3 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (3.0.2)
Requirement already satisfied: nbclient>=0.5.0 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (0.10.0)
Requirement already satisfied: nbformat>=5.7 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (5.10.4)
Requirement already satisfied: pandocfilters>=1.4.1 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (1.5.1)
Requirement already satisfied: tinycss2 in d:\code\ibmsb\.venv\lib\site-packages (from nbconvert->jupyter) (1.3.0)
Requirement already satisfied: jupyter-server<3,>=2.4.0 in d:\code\ibmsb\.venv\lib\site-packages (from notebook->jupyter) (2.14.1)
Requirement already satisfied: jupyterlab-server<3,>=2.27.1 in d:\code\ibmsb\.venv\lib\site-packages (from notebook->jupyter) (2.27.2)
Requirement already satisfied: jupyterlab<4.3,>=4.2.0 in d:\code\ibmsb\.venv\lib\site-packages (from notebook->jupyter) (4.2.3)
Requirement already satisfied: notebook-shim<0.3,>=0.2 in d:\code\ibmsb\.venv\lib\site-packages (from notebook->jupyter) (0.2.4)
Collecting qtpy>=2.4.0 (from qtconsole->jupyter)
  Downloading QtPy-2.4.1-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: six>=1.9.0 in d:\code\ibmsb\.venv\lib\site-packages (from bleach!=5.0.0->nbconvert->jupyter) (1.16.0)
Requirement already satisfied: webencodings in d:\code\ibmsb\.venv\lib\site-packages (from bleach!=5.0.0->nbconvert->jupyter) (0.5.1)
Requirement already satisfied: decorator in d:\code\ibmsb\.venv\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter) (5.1.1)
Requirement already satisfied: jedi>=0.16 in d:\code\ibmsb\.venv\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.19.1)
Requirement already satisfied: stack-data in d:\code\ibmsb\.venv\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.6.3)
Requirement already satisfied: colorama in d:\code\ibmsb\.venv\lib\site-packages (from ipython>=7.23.1->ipykernel->jupyter) (0.4.6)
Requirement already satisfied: python-dateutil>=2.8.2 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-client>=6.1.12->ipykernel->jupyter) (2.9.0.post0)
Requirement already satisfied: platformdirs>=2.5 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel->jupyter) (4.2.2)
Requirement already satisfied: pywin32>=300 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel->jupyter) (306)
Requirement already satisfied: anyio>=3.1.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (4.4.0)
Requirement already satisfied: argon2-cffi>=21.1 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (23.1.0)
Requirement already satisfied: jupyter-events>=0.9.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (0.10.0)
Requirement already satisfied: jupyter-server-terminals>=0.4.4 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (0.5.3)
Requirement already satisfied: overrides>=5.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (7.7.0)
Requirement already satisfied: prometheus-client>=0.9 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (0.20.0)
Requirement already satisfied: pywinpty>=2.0.1 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (2.0.13)
Requirement already satisfied: send2trash>=1.8.2 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (1.8.3)
Requirement already satisfied: terminado>=0.8.3 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (0.18.1)
Requirement already satisfied: websocket-client>=1.7 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-server<3,>=2.4.0->notebook->jupyter) (1.8.0)
Requirement already satisfied: async-lru>=1.0.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook->jupyter) (2.0.4)
Requirement already satisfied: httpx>=0.25.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook->jupyter) (0.27.0)
Requirement already satisfied: jupyter-lsp>=2.0.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook->jupyter) (2.2.5)
Requirement already satisfied: setuptools>=40.1.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab<4.3,>=4.2.0->notebook->jupyter) (70.2.0)
Requirement already satisfied: babel>=2.10 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook->jupyter) (2.15.0)
Requirement already satisfied: json5>=0.9.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook->jupyter) (0.9.25)
Requirement already satisfied: jsonschema>=4.18.0 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook->jupyter) (4.22.0)
Requirement already satisfied: requests>=2.31 in d:\code\ibmsb\.venv\lib\site-packages (from jupyterlab-server<3,>=2.27.1->notebook->jupyter) (2.32.3)
Requirement already satisfied: fastjsonschema>=2.15 in d:\code\ibmsb\.venv\lib\site-packages (from nbformat>=5.7->nbconvert->jupyter) (2.20.0)
Requirement already satisfied: wcwidth in d:\code\ibmsb\.venv\lib\site-packages (from prompt-toolkit>=3.0.30->jupyter-console->jupyter) (0.2.13)
Requirement already satisfied: soupsieve>1.2 in d:\code\ibmsb\.venv\lib\site-packages (from beautifulsoup4->nbconvert->jupyter) (2.5)
Requirement already satisfied: idna>=2.8 in d:\code\ibmsb\.venv\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (3.7)
Requirement already satisfied: sniffio>=1.1 in d:\code\ibmsb\.venv\lib\site-packages (from anyio>=3.1.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (1.3.1)
Requirement already satisfied: argon2-cffi-bindings in d:\code\ibmsb\.venv\lib\site-packages (from argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook->jupyter) (21.2.0)
Requirement already satisfied: certifi in d:\code\ibmsb\.venv\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook->jupyter) (2024.7.4)
Requirement already satisfied: httpcore==1.* in d:\code\ibmsb\.venv\lib\site-packages (from httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook->jupyter) (1.0.5)
Requirement already satisfied: h11<0.15,>=0.13 in d:\code\ibmsb\.venv\lib\site-packages (from httpcore==1.*->httpx>=0.25.0->jupyterlab<4.3,>=4.2.0->notebook->jupyter) (0.14.0)
Requirement already satisfied: parso<0.9.0,>=0.8.3 in d:\code\ibmsb\.venv\lib\site-packages (from jedi>=0.16->ipython>=7.23.1->ipykernel->jupyter) (0.8.4)
Requirement already satisfied: attrs>=22.2.0 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (23.2.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (2023.12.1)
Requirement already satisfied: referencing>=0.28.4 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (0.35.1)
Requirement already satisfied: rpds-py>=0.7.1 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema>=4.18.0->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (0.18.1)
Requirement already satisfied: python-json-logger>=2.0.4 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (2.0.7)
Requirement already satisfied: pyyaml>=5.3 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (6.0.1)
Requirement already satisfied: rfc3339-validator in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (0.1.4)
Requirement already satisfied: rfc3986-validator>=0.1.1 in d:\code\ibmsb\.venv\lib\site-packages (from jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (0.1.1)
Requirement already satisfied: charset-normalizer<4,>=2 in d:\code\ibmsb\.venv\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (3.3.2)
Requirement already satisfied: urllib3<3,>=1.21.1 in d:\code\ibmsb\.venv\lib\site-packages (from requests>=2.31->jupyterlab-server<3,>=2.27.1->notebook->jupyter) (2.2.2)
Requirement already satisfied: executing>=1.2.0 in d:\code\ibmsb\.venv\lib\site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (2.0.1)
Requirement already satisfied: asttokens>=2.1.0 in d:\code\ibmsb\.venv\lib\site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (2.4.1)
Requirement already satisfied: pure-eval in d:\code\ibmsb\.venv\lib\site-packages (from stack-data->ipython>=7.23.1->ipykernel->jupyter) (0.2.2)
Requirement already satisfied: fqdn in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (1.5.1)
Requirement already satisfied: isoduration in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (20.11.0)
Requirement already satisfied: jsonpointer>1.13 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (3.0.0)
Requirement already satisfied: uri-template in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (1.3.0)
Requirement already satisfied: webcolors>=1.11 in d:\code\ibmsb\.venv\lib\site-packages (from jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (24.6.0)
Requirement already satisfied: cffi>=1.0.1 in d:\code\ibmsb\.venv\lib\site-packages (from argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook->jupyter) (1.16.0)
Requirement already satisfied: pycparser in d:\code\ibmsb\.venv\lib\site-packages (from cffi>=1.0.1->argon2-cffi-bindings->argon2-cffi>=21.1->jupyter-server<3,>=2.4.0->notebook->jupyter) (2.22)
Requirement already satisfied: arrow>=0.15.0 in d:\code\ibmsb\.venv\lib\site-packages (from isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (1.3.0)
Requirement already satisfied: types-python-dateutil>=2.8.10 in d:\code\ibmsb\.venv\lib\site-packages (from arrow>=0.15.0->isoduration->jsonschema[format-nongpl]>=4.18.0->jupyter-events>=0.9.0->jupyter-server<3,>=2.4.0->notebook->jupyter) (2.9.0.20240316)
Downloading jupyter-1.0.0-py2.py3-none-any.whl (2.7 kB)
Downloading ipywidgets-8.1.3-py3-none-any.whl (139 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 139.4/139.4 kB 4.2 MB/s eta 0:00:00
Downloading jupyter_console-6.6.3-py3-none-any.whl (24 kB)
Downloading qtconsole-5.5.2-py3-none-any.whl (123 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 123.4/123.4 kB 7.1 MB/s eta 0:00:00
Downloading jupyterlab_widgets-3.0.11-py3-none-any.whl (214 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 214.4/214.4 kB 12.8 MB/s eta 0:00:00
Downloading QtPy-2.4.1-py3-none-any.whl (93 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 93.5/93.5 kB ? eta 0:00:00
Downloading widgetsnbextension-4.0.11-py3-none-any.whl (2.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.3/2.3 MB 25.0 MB/s eta 0:00:00
Installing collected packages: widgetsnbextension, qtpy, jupyterlab-widgets, ipywidgets, qtconsole, jupyter-console, jupyter
Successfully installed ipywidgets-8.1.3 jupyter-1.0.0 jupyter-console-6.6.3 jupyterlab-widgets-3.0.11 qtconsole-5.5.2 qtpy-2.4.1 widgetsnbextension-4.0.11

[notice] A new release of pip is available: 24.1.1 -> 24.1.2
[notice] To update, run: python.exe -m pip install --upgrade pip
</details>

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

##### 6.3 List current kernels

```python
jupyter kernelspec list

```

#### Jupyter Kernelspec

```plaintext
jupyter kernelspec
No subcommand specified. Must specify one of: ['list', 'install', 'uninstall', 'remove', 'install-self', 'provisioners']  

Manage Jupyter kernel specifications.

Subcommands
===========
Subcommands are launched as `jupyter kernelspec cmd [args]`. For information on
using subcommand 'cmd', do: `jupyter kernelspec cmd -h`.

list
    List installed kernel specifications.
install
    Install a kernel specification directory.
uninstall
    Alias for remove
remove
    Remove one or more Jupyter kernelspecs by name.
    List available provisioners for use in kernel specifications.

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