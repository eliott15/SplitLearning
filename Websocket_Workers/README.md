# Websocket Workers 

Start the workers by running 
``` 
 $ python start_websocket_server 
```
This script will create three websocket workers, on three different port. 
Then, run 
```
$ python3 run_websocket_client_expname
```
where expname is the name of the experiment (SL, fedl, or favg).
The script will distribute MNIST dataset through the workers, and train a CNN according to the selected method.
