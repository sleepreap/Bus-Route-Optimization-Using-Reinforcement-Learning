import requests
import json

# API endpoint
url_bus_routes= "http://datamall2.mytransport.sg/ltaodataservice/BusRoutes"
url_bus_stops = "http://datamall2.mytransport.sg/ltaodataservice/BusStops"
# Headers for the API request
headers = {
    'AccountKey': ''
}

# Initialize dictionary to store results
bus_routes = {}
bus_stops = {}

# Pagination parameters
skip = 0
top = 500  # Number of records to fetch per request

while True:
    # Make the API call with pagination
    params = {
        '$skip': skip,
        '$top': top
    }
    response = requests.get(url_bus_routes, headers=headers, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response

        # Process each bus stop entry in the JSON
        for entry in data['value']:
            service_no = entry['ServiceNo']
            bus_stop_code = entry['BusStopCode']
            distance = entry['Distance']
            
            # Initialize the service number key if it doesn't exist
            if service_no not in bus_routes:
                bus_routes[service_no] = []

            # Append the bus stop code and distance to the service number list
            bus_routes[service_no].append([distance, bus_stop_code])

        # Check if we've received fewer results than 'top', which means it's the last page
        if len(data['value']) < top:
            break

        # Update the skip value for the next page
        skip += top
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        break

# Recalculate distances between bus stops and round to 1 decimal place
for service_no, stops in bus_routes.items():
    for i in range(len(stops) - 1, 0, -1):
        stops[i][0] = round(stops[i][0] - stops[i - 1][0], 1)

# Print the final result

skip = 0
top = 500

# Fetch bus stops data
while True:
    # Make the API call with pagination
    params = {
        '$skip': skip,
        '$top': top
    }
    response = requests.get(url_bus_stops, headers=headers, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        data = response.json()  # Parse JSON response

        # Process each bus stop entry in the JSON
        for entry in data['value']:
            bus_stop_code = entry['BusStopCode']
            latitude = entry['Latitude']
            longitude = entry['Longitude']
            
            # Add the bus stop code and its [latitude, longitude] to the dictionary
            bus_stops[bus_stop_code] = [latitude, longitude]

        # Check if we've received fewer results than 'top', which means it's the last page
        if len(data['value']) < top:
            break

        # Increment the counter for the next page
        skip += top
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        break

# Append latitude and longitude to bus_routes
for service_no, stops in bus_routes.items():
    for stop in stops:
        bus_stop_code = stop[1]
        if bus_stop_code in bus_stops:
            stop.extend(bus_stops[bus_stop_code])

# # Print the updated bus_routes dictionary
# print(bus_routes)

# Save the bus_routes dictionary to a JSON file
with open('bus_routes_dict.json', 'w') as f:
    json.dump(bus_routes, f, indent=4)