# ConcentrIoT
#### an open-source monitoring and management protocol for IoT and Smart Home Appliances

## Idea
This protocol is meant to be used with IoT and Smart Home Appliances in order to provide additional functionality, both in the local network and over the Internet, offering a FOSS replacement for solutions like Azure IoT Hub. Currently, however, it is just an idea that needs to be implemented in real-world devices. 

ConcentrIoT offers a number of functionalities, thanks to its semi-centralised infrastructure, with *the HUB* device being responsible for providing communication layer between devices, unifying the settings panel, as well as governing the available resources (for example, if one of the devices needs to download a large resource package, or train a neural network).

### The HUB device
The HUB device is the center  point for the infrastructure. Any Linux-capable machine can become a HUB, including Raspberry Pi, BeagleBone, or even your old computer!

## Message format
The protocol works in the 5th layer of the OSI Reference Model. Its message format is simmilar to the one of SIP, with the removal of unnecessary fields.

<table width="600px">
    <tbody>
        <tr>
            <td width="100px" align=center valign=center>Operation</td>
            <td width="100px" align=center valign=center>URI</td>
            <td width="100px" align=center valign=center>Version</td>
        </tr>
        <tr>
            <td align=center valign=center colspan=3>Sender: <i>Sender IP, URI or alias</i></td>
        </tr>
        <tr>
            <td align=center valign=center colspan=3>Receiver: <i>Receiver IP, URI or alias</i></td>
        </tr>
        <tr>
            <td align=center valign=center colspan=3>Content-Type: <i>MIME Content Type</i></td>
        </tr>
        <tr>
            <td align=center valign=center colspan=3>Content-Length: <i>Length</i></td>
        </tr>
        <tr>
        	<td align=center valign=center colspan=3 rowspan=3>
        		key1='<i>val1</i>', key2='<i>val2</i>', ... <br>
        		keyN='<i>valN</i>'
        	</td>
        </tr>
    </tbody>
</table>

### Available *request* operations 
- *DISCOVER* - sent on a regular basis in order to keep up the neighbourhood relationship
- *FILE* - announce a file transfer (required parameters: fname, size, type, importance)
- *REQFILE* - request a file transfer (required parameters: fname, type, importance)
- *COMPUTE* - request a computation (required parameters: importance, attachments, cmd, required-packages)
- *AUTHDATA* - request authentication with data (required parameters: challenge-method, challenge parameters)
- *CONFIGUPDATE* - update configuration (required parameters: config - stringified JSON configuration inside)
- *CONFIGGET* - get a configuration dump

### Available *response* operations
- *ACK* - confirmation after *DISCOVER*, *FILE* (also sent when file has been received), *REQFILE* if a file exists
- *NACK* - negative response, can be used in the place of *ACK* if an error has occured. In case of *FILE* should result in an retransmission.
- *AUTHENTICATE* - authentication needed (required parameters: challenge-method, challenge parameters)
- *CONFIGCONTENT* - contains a JSON dump of the configuration (required parameters: config - stringified JSON configuration inside)
- *ERROR* - a fatal error has occured (required parameters: ecode - error code)

### Data Generator
The data generator is intended to perform extremely deep augmentation to force NN to recognize robust features from a small dataset. In our demo, we are training NN to recognize our car. To achieve the satisfactory performance we generate data samples under lazy evaluation. Samples of our car are merged with the random background. All samples are transferred through geometric and color augmentation. In a training loop, data are fetched under lazy evaluation. Working with such architecture, we can parallelize data generation as images in a batch can be generated parallelly. With an optimized input pipeline, we were able to reduce the time of learning four times. Accuracy is very close to 100%, nevertheless, due to the small set of photos of our car, we cannot rely on validation probability. However, in practice, the networks well when exposed to the images captured from the camera. 

### Data Generator Optimization
Dataset pipelines are bottlenecks in many ML applications that rely on dynamically generate the database. Therefore, we should take care of it as it's a place where we should start optimizing our ML workflow. Firstly, the independence of sample generation allows us to benefit from parallelizing data extraction. To mitigate data extraction we use tf.data.Dataset.interleav transformation to parallelize the data loading step. Then, we can cach a dataset in memory to save some operations. Finally, we can save a lot of time prefetching overlaps of the preprocessing and model execution of a training step. Overall, we reduced the time of learning four times. Below, we can see the results of our optimizations. 
#### Naive data pipeline
![alt text](https://github.com/kondziusob/ConcentrIoT/raw/master/naive.jpg)

#### Optimized data pipeline
![alt text](https://github.com/kondziusob/ConcentrIoT/raw/master/optimized.jpg)

