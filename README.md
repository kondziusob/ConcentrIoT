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
Data generator is intended to perform extremelly deep augmentation to force NN to recognize robust features from small dataset. In our demo, we are training NN to recognize our car. To achieve satisfaiable performance we generate datasaples under lazy evaluation. Samples of our car are merged with random background. All samples are transfered through geometric and colour augmentation. In a training loop, data are fatched under lazy evaluation (we fatch data only when data generator get request to do so). Working with such architecture, we can paralelize data generation as imageas in a batch can be generate parallely. With optimized input pipeline, we were able to reduce the time of learning four times. Accuracy, is very close to 100%, nevertheless due to the small set of photos of our car, we cannot relay on validation probability. However, in practise, the net works well when exposed to the images captured form camera. 
