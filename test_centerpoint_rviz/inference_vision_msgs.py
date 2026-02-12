import sys
sys.modules['coverage'] = None
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
import numpy as np
import torch
from mmdet3d.apis import init_model, inference_detector
from mmengine.runner import load_checkpoint
import glob
import time
import threading
import math

torch.backends.cudnn.benchmark = True

def pointcloud2_to_numpy(msg):
    dtype_list = [
        ('x', np.float32), 
        ('y', np.float32), 
        ('z', np.float32), 
        ('intensity', np.float32)
    ]
    
    if msg.point_step > 16:
        dtype_list.append(('padding', np.uint8, (msg.point_step - 16)))
        
    try:
        data = np.frombuffer(msg.data, dtype=np.dtype(dtype_list))
    except ValueError:
        return np.zeros((0, 4), dtype=np.float32)

    points = np.column_stack((data['x'], data['y'], data['z'], data['intensity'])).astype(np.float32).copy()
    return points

class CenterPointNode(Node):
    def __init__(self):
        super().__init__('centerpoint_node')
        self.lidar_topic = '/carla/hero/lidar'

        try:
            self.config = glob.glob("centerpoint*.py")[0]
            self.checkpoint = glob.glob("centerpoint*.pth")[0]
        except IndexError:
            self.get_logger().error("File not found")
            return

        self.model = init_model(self.config, checkpoint=None, device='cuda:0')
        
        try:
            load_checkpoint(self.model, self.checkpoint, map_location='cuda:0', strict=False)
        except Exception as e:
            pass

        try:
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(self.model)
        except:
            pass

        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.listener_callback, 1)
        self.box_pub = self.create_publisher(Detection3DArray, '/detected_objects', 1)
        
        self.latest_msg = None
        self.msg_lock = threading.Lock()
        self.running = True
        
        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()

    def listener_callback(self, msg):
        with self.msg_lock:
            self.latest_msg = msg

    def inference_loop(self): 
        while self.running and rclpy.ok():
            msg = None
            with self.msg_lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None 
            
            if msg is not None:
                self.process_msg(msg)
            else:
                time.sleep(0.001)

    def process_msg(self, msg):
        t0 = time.time()
        
        points = pointcloud2_to_numpy(msg)
        if points.shape[0] == 0: return

        points = points[::4]
        if points[:, 3].max() > 1.0:
            points[:, 3] /= 255.0

        times = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, times))

        torch.cuda.synchronize()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result, data = inference_detector(self.model, points)
        torch.cuda.synchronize()

        pred_inst = result.pred_instances_3d
        bboxes = pred_inst.bboxes_3d.tensor.cpu().numpy()
        scores = pred_inst.scores_3d.cpu().numpy()
        labels = pred_inst.labels_3d.cpu().numpy()
        
        detection_array = Detection3DArray()
        detection_array.header = msg.header
        
        for i, bbox in enumerate(bboxes):
            if scores[i] < 0.5: continue
            
            cx, cy, cz, l, w, h, rot = bbox[:7]
            label_idx = int(labels[i])
            score = float(scores[i])
            
            detection = Detection3D()
            detection.header = msg.header
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(label_idx)
            hypothesis.hypothesis.score = score
            detection.results.append(hypothesis)
            
            detection.bbox.center.position.x = float(cx)
            detection.bbox.center.position.y = float(cy)
            detection.bbox.center.position.z = float(cz)
            detection.bbox.center.orientation.x = 0.0
            detection.bbox.center.orientation.y = 0.0
            detection.bbox.center.orientation.z = math.sin(rot / 2.0)
            detection.bbox.center.orientation.w = math.cos(rot / 2.0)
            detection.bbox.size.x = float(l)
            detection.bbox.size.y = float(w)
            detection.bbox.size.z = float(h)
            
            detection_array.detections.append(detection)
            
        self.box_pub.publish(detection_array)

    def destroy_node(self):
        self.running = False
        self.inference_thread.join()
        super().destroy_node()

def main():
    rclpy.init()
    node = CenterPointNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()