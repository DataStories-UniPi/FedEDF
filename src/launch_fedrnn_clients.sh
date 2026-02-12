#!/bin/bash

# ALWAYS SET IT FIRST -> param. name: "federation"
data_name="$2"  #  dundee | porto

# Define the list of available EVSE Hubs
if [[ "$data_name" == "dundee" ]]
then
    echo "Launching Dundee Dataset Federation..."
    cluster_ids=(0 1 2 3 4 5 6 7)

elif [[ "$data_name" == "porto" ]]
then
    echo "Launching Porto Dataset Federation"
    cluster_ids=(0 1 2 3)

elif [[ "$data_name" == "boulder" ]]
then
    echo "Launching Boulder Dataset Federation"
    cluster_ids=(0 1 2 3 4 5 6 7)

elif [[ "$data_name" == "paloalto" ]]
then
    echo "Launching Palo Alto Dataset Federation"
    cluster_ids=(0 1 2 3 4 5 6 7)
fi

# Iterate through each item in the list
for cluster_id in "${cluster_ids[@]}"; do
    python rnn_client.evse_hub.py --cluster_id $cluster_id "$@" &
done

# Wait for all background jobs to finish
wait

echo "All processes have finished!"
