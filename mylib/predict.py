from mylib import api

def predict(client, name, names, values):
    id = api.deployments["model_" + name]
    input_data = {
        "input_data": 
        [
            {
                'fields': names, 
                'values': [
                    values
                ]
            }
        ]
    }
    predictions = client.deployments.score(id, input_data)
    return predictions["predictions"][0]["values"][0][0]