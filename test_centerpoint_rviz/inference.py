import sys
sys.modules['coverage'] = None
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import torch
from mmdet3d.apis import init_model, inference_detector
from mmengine.runner import load_checkpoint
import os
import glob
import time
import threading

torch.backends.cudnn.benchmark = True

def pointcloud2_to_numpy(msg): # PointCloud2 -> Numpy 변환
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
        super().__init__('centerpoint_node') # 노드 이름 설정
        self.lidar_topic = '/carla/hero/lidar' # 구독할 라이다 토픽

        try:
            self.config = glob.glob("centerpoint*.py")[0]
            self.checkpoint = glob.glob("centerpoint*.pth")[0]
        except IndexError:
            self.get_logger().error("File not found")
            return

        self.get_logger().info(f'Config: {self.config}')
        self.get_logger().info(f'Checkpoint: {self.checkpoint}')
        
        self.model = init_model(self.config, checkpoint=None, device='cuda:0')
        
        try:
            load_checkpoint(self.model, self.checkpoint, map_location='cuda:0', strict=False)
        except Exception as e:
            self.get_logger().error(f"Load failed: {e}")

        try:
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(self.model)
        except:
            pass

        # 구독 및 발행
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.listener_callback, 1)
        self.pub = self.create_publisher(MarkerArray, '/detection_markers', 1)
        
        self.latest_msg = None
        self.msg_lock = threading.Lock()
        self.running = True
        
        self.colors = [
            (0.0, 1.0, 0.0), # car
            (0.0, 1.0, 1.0), # truck   
            (1.0, 1.0, 0.0), # construction_vehicle
            (0.0, 0.0, 1.0), # bus
            (0.5, 0.0, 0.5), # trailer
            (0.8, 0.8, 0.8), # barrier
            (1.0, 0.5, 0.0), # motorcycle
            (1.0, 0.0, 0.0), # bicycle
            (1.0, 0.0, 1.0), # pedestrian
            (1.0, 0.6, 0.0)  # traffic_cone
        ]

        self.inference_thread = threading.Thread(target=self.inference_loop)
        self.inference_thread.start()
        
        self.get_logger().info('Ready to Inference')

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

    def process_msg(self, msg): # 처리
        t0 = time.time()
        
        points = pointcloud2_to_numpy(msg) # 변환

        if points.shape[0] == 0: return

        points = points[::4]

        if points[:, 3].max() > 1.0:
            points[:, 3] /= 255.0

        times = np.zeros((points.shape[0], 1), dtype=np.float32)
        points = np.hstack((points, times))

        t1 = time.time()

        torch.cuda.synchronize()
        t_model_start = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result, data = inference_detector(self.model, points)
        
        torch.cuda.synchronize()
        t_model_end = time.time()

        pred_inst = result.pred_instances_3d
        bboxes = pred_inst.bboxes_3d.tensor.cpu().numpy()
        scores = pred_inst.scores_3d.cpu().numpy()
        labels = pred_inst.labels_3d.cpu().numpy()
        
        markers = MarkerArray() # 마커 초기화
        markers.markers.append(Marker(action=Marker.DELETEALL))

        detected_count = 0
        
        for i, bbox in enumerate(bboxes):
            if scores[i] < 0.5: continue
            
            detected_count += 1
            cx, cy, cz, l, w, h, rot = bbox[:7]
            label_idx = int(labels[i])
            
            marker = Marker()
            marker.header = msg.header 
            marker.ns = "boxes"
            marker.id = i
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0 
            
            marker.scale.x = 0.05 
            
            if 0 <= label_idx < len(self.colors):
                r, g, b = self.colors[label_idx]
                marker.color.r = float(r)
                marker.color.g = float(g)
                marker.color.b = float(b)
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 1.0
                
            marker.color.a = 1.0 
            
            cos_r = np.cos(rot)
            sin_r = np.sin(rot)
            
            cz = cz + h/2 

            corners = np.array([
                [ l/2,  w/2,  h/2], [ l/2, -w/2,  h/2], [-l/2, -w/2,  h/2], [-l/2,  w/2,  h/2],
                [ l/2,  w/2, -h/2], [ l/2, -w/2, -h/2], [-l/2, -w/2, -h/2], [-l/2,  w/2, -h/2]
            ])

            rot_corners = np.copy(corners)
            rot_corners[:, 0] = corners[:, 0] * cos_r - corners[:, 1] * sin_r + cx
            rot_corners[:, 1] = corners[:, 0] * sin_r + corners[:, 1] * cos_r + cy
            rot_corners[:, 2] = corners[:, 2] + cz

            lines_indices = [
                0,1, 1,2, 2,3, 3,0,
                4,5, 5,6, 6,7, 7,4,
                0,4, 1,5, 2,6, 3,7
            ]

            for idx in lines_indices:
                p = Point()
                p.x, p.y, p.z = float(rot_corners[idx, 0]), float(rot_corners[idx, 1]), float(rot_corners[idx, 2])
                marker.points.append(p)

            markers.markers.append(marker)
            
        self.pub.publish(markers)
        t_end = time.time()
        
        prep_time = (t1 - t0) * 1000
        infer_time = (t_model_end - t_model_start) * 1000
        post_time = (t_end - t_model_end) * 1000

        time_total = prep_time + infer_time + post_time
        total_fps = 1.0 / (t_end - t0)

        print(f"[Time: {time_total:.2f}ms] | FPS: {total_fps:.2f} | Boxes: {detected_count}")

    def destroy_node(self):
        self.running = False
        self.inference_thread.join()
        super().destroy_node()

def main():
    rclpy.init()
    node = CenterPointNode()
    try:
        rclpy.spin(node) # 노드 실행
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown() # ROS 종료

if __name__ == '__main__':
    main()