# Simmonitor
Generate reports for Einstein Toolkit Simulations
## Setup
- Setup a base `texlive` install along with the `cm-super` and `ams-math` package
```sh
git clone https://github.com/chir4gm/simmonitor
cd simmonitor
# For just static simulation reports
pip3 install -r requirements_2d.txt
# For both static and interactive reports
pip3 install -r requirements_3d.txt
```
## Usage
- Use `simmonitor_[2d|3d].cfg` or pass parameters to the script as command-line arguments

### 2D
Create static report in `./output`
```sh
./simmonitor_2d.py -c simmonitor_2d.cfg --datadir [simulation directory] -v
```
### 3D
```sh
./simmonitor_2d.py -c simmonitor_3d.cfg --datadir [simulation directory] --output [output file] -v
```