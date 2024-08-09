import folium
import json

# Load the JSON data
with open('bus_routes_dict.json', 'r') as f:
    bus_routes = json.load(f)
    
# Replace with the bus service numbers you want to extract
bus_service_numbers = ['99', '185']  
included_services = {service_no: bus_routes[service_no] for service_no in bus_service_numbers if service_no in bus_routes}

# Create a base map of Singapore
singapore_map = folium.Map(location=[1.3521, 103.8198], zoom_start=12)

# Add a route to the map with a specific color
def add_route_to_map(service_no, stops):
    feature_group = folium.FeatureGroup(name=f'Service {service_no}')
    
    # Add nodes (bus stops) to the map
    for i in range(len(stops)):
        distance, bus_stop_code, lat, lon = stops[i]
        
        # Add a marker for the bus stop
        folium.Marker(
            location=[lat, lon],
            popup=f'Stop {bus_stop_code}<br>Distance: {distance} km',
            icon=folium.Icon()
        ).add_to(feature_group)
        
        # Draw a line to the next stop if it exists
        if i < len(stops) - 1:
            next_distance, _, next_lat, next_lon = stops[i + 1]
            folium.PolyLine(
                locations=[[lat, lon], [next_lat, next_lon]],
                weight=2,
                opacity=0.7
            ).add_to(feature_group)
    
    feature_group.add_to(singapore_map)

# Add all bus routes to the map
for service_no, stops in included_services.items():
    add_route_to_map(service_no, stops)
    

# Save the map to an HTML file
singapore_map.save('singapore_map.html')

print("Map has been saved to 'singapore_map.html'.")
