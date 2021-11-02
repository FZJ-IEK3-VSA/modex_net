<a href="https://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="https://www.fz-juelich.de/SharedDocs/Bilder/IBG/IBG-3/DE/Plant-soil-atmosphere%20exchange%20processes/INPLAMINT%20(BONARES)/Bild3.jpg?__blob=poster" alt="Forschungszentrum Juelich Logo" width="230px"></a> 

<a href="https://www.energiesystem-forschung.de/forschen/projekte/modex-net"><img src="https://www.energiesystem-forschung.de/lw_resource/datapool/systemfiles/cbox/1414/live/lw_bild/modexnet_logo.png" alt="Logo Modex Net" width="150px"></a>

# modex_net
modex_net is a python package for producing indicators and visualizations of market results for various power system models as uploaded in the [Open Energy Platform](https://openenergy-platform.org). The package was created in the context of the [MODEX-Net project](https://www.energiesystem-forschung.de/forschen/projekte/modex-net) funded by the German Ministry for Economic Affairs and Energy. 

## Installation
Clone this repository using 
```
git clone https://github.com/FZJ-IEK3-VSA/modex_net.git
```

It is recommended to also create a separate virtual environment before the installation via
```
python -m venv modex_net_env
.\modex_net_env\Scripts\activate
```
for windows or
```
python3 -m venv modex_net_env
source modex_net_env/bin/activate
```
for unix/macOS or alternatively with Anaconda:
```
conda create --file requirements.yml
conda activate modex_net_env
```

Install via pip using
```
cd modex_net
python -m pip install .
```
or via python using
```
python setup.py install
```
## Examples
A short guide with examples can be found in the examples folder as a jupyter notebook. For that you will need to also install jupyter in your environment.
```
python -m pip install jupyter
```
or
```
python3 -m pip install jupyter
```
for unix/macOS
or
```
conda install jupyter
```
with Anaconda
## Testing
For developers, you can use the requirements-dev.yml for the installation, that include pytest. When testing you will need to add your personal API token for the Open Energy Platform via
```
pytest --token <OEP_TOKEN>
```
## License
This package uses GNU GPLv3, since this is the one used by the ntaylor package that is included.

## About Us 
<a href="http://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html"><img src="https://www.fz-juelich.de/SharedDocs/Bilder/IEK/IEK-3/Abteilungen2015/VSA_DepartmentPicture_2019-02-04_459x244_2480x1317.jpg?__blob=normal" width="400px" alt="Abteilung VSA"></a> 

We are the [Techno-Economic Energy Systems Analysis](https://www.fz-juelich.de/iek/iek-3/EN/Forschung/_Process-and-System-Analysis/_node.html) department at the [Institute of Energy and Climate Research: Electrochemical Process Engineering (IEK-3)](https://www.fz-juelich.de/iek/iek-3/EN/Home/home_node.html) belonging to the [Forschungszentrum Jülich](https://www.fz-juelich.de/). Our interdisciplinary department's research is focusing on energy-related process and systems analyses. Data searches and system simulations are used to determine energy and mass balances, as well as to evaluate performance, emissions and costs of energy systems. The results are used for performing comparative assessment studies between the various systems. Our current priorities include the development of energy strategies, in accordance with the German Federal Government’s greenhouse gas reduction targets, by designing new infrastructures for sustainable and secure energy supply chains and by conducting cost analysis studies for integrating new technologies into future energy market frameworks.

## Further reading
Three publications are under revision at the MODEX special issue of the Renewable & Sustainable Energy Reviews journal by Elsevier

## Acknowledgements
We would like to thank all partners and project members for their contributions, especially
- German Aerospace Center (DLR), Institute of Networked Energy Systems: Oriol Raventós, Chinonso Unaichi, Julian Bartels, Jan Buschmann, Wided Medjroubi
- Forschungsstelle für Energiewirtschaft (FfE): Andreas Bruckmeier, Timo Kern, Felix Böing, Tobias Schmid, Christoph Pellinger
- Karlsruhe Institute of Technology (KIT), Institute for Industrial Production (IIP): Thomas Dengiz, Rafael Finck, Armin Ardone, Katrin Seddig, Manuel Ruppert
- Öko-Institut e.V., Energy & Climate Division: Matthias Koch, Christian Winger, Franziska Flachsbarth, Sebastian Palacios and Susanne Krieger
- RWTH Aachen University, High Voltage Equipment and Grids, Digitalization and Energy Economics (IAEW): Jonas Mehlem, Lukas Weber, Annika Klette, Levin Skiba, Alexander Fehler 
- TU Dortmund University, Institute of Energy Systems, Energy Efficiency and Energy Economics (ie3): Björn Matthes, Jan Peper
- Technische Universität Dresden, Chair of Energy Economics: Hannes Hobbie, Christina Wolff, David Schönheit, Dominik Möst


This work was supported by the Federal Ministry for Economic Affairs and Energy in the 6. energy research funding program, grant number 03ET4074.

<a href="https://www.energiesystem-forschung.de/forschen/projekte/modex-net"><img src="https://www.energiesystem-forschung.de/lw_resource/layoutfiles/img/BMWi_Office_Farbe_de_WBZ.jpg" alt="Bundesministerium für Wirtschaft und Energie" width="200px"></a>
