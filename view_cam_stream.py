import requests
import cv2
import numpy as np
import datetime
from loguru import logger
import click

logger.info("Start reading from the stream of mouth ai camera...")

@click.command()
@click.option('--url', prompt='ai mouth service endpoint', default='http://192.168.1.144:10960/get_frame')
def view_stream(url):
    header = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
    # call the API and iterate over the frames returned
    response = requests.get(url, stream=True)
    if response.ok:
        data = b''
        for chunk in response.iter_content(chunk_size=1024):
            if not chunk:
                logger.error("no chunk found in response")
                break
            data += chunk
            split_res = data.split(header)
            if len(split_res) == 3:
                data = header + split_res[2]
                byte_frame = split_res[1]
                nparr = np.frombuffer(byte_frame, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # add timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame size: {frame.shape[1]}x{frame.shape[0]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # draw center point
                center = (frame.shape[1]//2, frame.shape[0]//2)
                cv2.circle(frame, center, 4, (0, 0, 255), -1)
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    view_stream()
