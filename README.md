Replicates original simmonitor tool. Enhanced with `bokeh` and `kuibit`.

Outputs a static `index.html`, along with a `interactive.html` which uses `bokeh`

## Prerequisites
- Setup a base `texlive` install along with the `cm-super` and `ams-math` package
```sh
git clone https://github.com/chir4gm/simmonitor
cd simmonitor
pip3 install -r requirements.txt
```

## Usage
- Use `simmonitor.cfg` or pass parameters to the script as command-line arguments

```sh
python3 script.py -c simmonitor.cfg
```
