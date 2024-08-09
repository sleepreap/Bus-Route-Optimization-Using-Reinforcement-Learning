# Bus Route Optimization Using Reinforcement Learning (Q-learning)
This project demonstrates reinforcement learning, q-learning, applied to bus route optimization within a specific area in Singapore. The goal is to find the most efficient bus routes between a starting location and a destination, considering both bus travel and walking distances. This functions similarly to how Google Maps would recommend which bus to take if you want to travel from one destination to another.

The bus stops are treated as nodes with the distance between them acting as weights for the edges connecting them. For reinforcement learning to work, additional edges are added between each bus stop. Edges are added when the actual distance between two bus stops is less than 1km. This allows reinforcement to learn multiple combinations of travel from one destination to another by taking different bus services via walking from one bus stop to another. 1KM is used as a benchmark of walkable distance. The distance is calculated using the lat long values of the bus stops.  To minimize the walking that one has to make for a bus route,  the weights connecting one bus stop to another via walking is the actual distance multiplied by 4, forcing the reinforcement learning to not favor such travel methods. The starting point is connected to any of the bus stops if the actual distance between them is less than 1km. 
The starting node is always treated as 0, destination node is always treated as 1.

## Requirements

- AccountKey for the LTA API

### packages:
- numpy
- networkx
- matplotlib
- folium (To see the bus stop nodes on the map)
- imageio (for generating the video/gif file to visualize)
- imageio-ffmpeg ( for generating the video/gif file to visualize)

To install them, try:

`Pip install networkx matplotlib imageio imageio-ffmpeg folium`


## APIs Used
1. LTA DataMall API - Bus Routes
- Endpoint: http://datamall2.mytransport.sg/ltaodataservice/BusRoutes
- Purpose: Retrieves detailed route information for all bus services currently in operation, including all bus stops along each route and first/last bus timings for each stop.
- Note: API responses are limited to 500 records per call. Use the $skip operator to retrieve subsequent records.
- Data Used: Bus stop codes and distances (used as weights for reinforcement learning).
  
2. LTA DataMall API - Bus Stops
- Endpoint: http://datamall2.mytransport.sg/ltaodataservice/BusStops
- Purpose: Retrieves detailed information for all bus stops currently serviced by buses, including bus stop code and location coordinates.
- Data Used: Latitude and longitude values for bus stops, which are matched with bus stop codes from the Bus Routes API.

3. OneMap API - Postal Code to Lat/Long Conversion
-Endpoint: https://www.onemap.gov.sg/api/common/elastic/search?searchVal=200640&returnGeom=Y&getAddrDetails=Y&pageNum=1
-Purpose: Converts a postal code to latitude and longitude values, used to determine the user's current location and destination.

Firstly, the bus routes API is used to get all bus stops for each bus service. For each bus service,  the bus stop code and distance between each bus stop are extracted. Since the distance is defined as the distance from the starting bus stop, preprocessing is done to get the actual distance from the adjacent bus stops. The bus stops API is then used to get all the latitude and longitude values for each of the bus stops. This is done by using the bus stop code and getting the corresponding latitude and longitude values.
A dictionary is used as a method of storage. The key would be the bus service.  Each bus service will have a list of bus stops in the following format [distance, bus stop code, lat, long ]. The dictionary is then saved as a JSON.  

## Python Scripts
`apicall.py`
This script handles the API calls to retrieve data on the bus routes and bus stops, then the data is stored as a JSON.

`map.py`
This script generates a map of Singapore with all the bus stops for the selected bus services plotted as nodes. The map can be accessed by opening the HTML file that will be created.

`RL.py`
This script demonstrates reinforcement learning to find the best bus route between a starting point and a destination in Singapore. The script would require user inputs for the starting destination and ending destination. The script also requires the user to define the  bus services used in the bus_service_numbers variable. The script can also set parameters such as the number of epochs. An image of a graph displaying the nodes and edges used for reinforcement learning will be displayed using Matplotlib, which must be closed. Then, Matplotlib will display reinforcement learning via q-learning. After closing the Matplotlib, a GIF of the q learning will be saved, and the output of the bus route to take will be produced. 

## Examples

1. Area of Focus: Jurong West area, with the destination set to Jurong Point Bus Interchange. Bus Services Considered: Limited to ["243G", "243W", "179"] due to computational constraints.
   
Starting destination postal code: 640835

Destination postal code: 648886

Epochs: 100

![Jurong Point Bus Interchange Graph](/figs_and_gifs/jurongpoint_graph.png)
 
![Jurong Point Bus Interchange Map](/figs_and_gifs/jurongpoint_map.PNG)

![Jurong Point Bus Interchange](/figs_and_gifs/jurongpoint_RL.gif)

![Output](/figs_and_gifs/jurongpoint_route.PNG)
   

   
3. Area of Focus: Jurong West area, with the destination set to Clementi Bus Interchange. Bus Services Considered: Limited to ["99", "185"] due to computational constraints.
   
Starting destination postal code: 640835
   
Destination postal code: 120441

Epochs: 100

![Clementi Bus Interchange Graph](/figs_and_gifs/clementi_graph.png)

![Clementi Bus Interchange Map](/figs_and_gifs/clementi_map.PNG)

![Clementi Bus Interchange](/figs_and_gifs/clementi_RL.gif)

![Output](/figs_and_gifs/clementi_route.PNG)

