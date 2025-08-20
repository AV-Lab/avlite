import torch
import numpy as np
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import datetime
# from c10_perception.c11_perception_model import PerceptionModel

import logging
log = logging.getLogger("c10_perception.c12_perception_strategy")
def calculate_ade(predictions, targets,per_sample: bool = False):
    """
    Compute Average Displacement Error (ADE) as the mean Euclidean distance 
    between the predicted and target positions over all time steps.
    
    This function converts inputs to tensors if they are provided as lists or NumPy arrays.
    
    Args:
        predictions (torch.Tensor or list or np.ndarray): Predicted positions with shape (B, T, 2)
            or (B, T, N, 2). If a list, each element is assumed to be a tensor (or convertible)
            with shape (T, N_valid, 2).
        targets (torch.Tensor or list or np.ndarray): Ground-truth positions with the same shape as predictions.

        per_sample (bool): If True, returns a NumPy array of per-sample ADE values.
            If False (default), returns a single scalar averaged over all samples.
    
    
    Returns:
        float: The average ADE computed over all time steps and nodes.
    """
    if isinstance(predictions, list):
        total_error = 0.0
        count = 0
        per_sample_errors = []
        for pred, targ in zip(predictions, targets):
            if not isinstance(pred, torch.Tensor):
                if isinstance(pred, np.ndarray):
                    pred = torch.from_numpy(pred)
                else:
                    pred = torch.tensor(pred)
            if not isinstance(targ, torch.Tensor):
                if isinstance(targ, np.ndarray):
                    targ = torch.from_numpy(targ)
                else:
                    targ = torch.tensor(targ)

            diff = torch.norm(pred - targ, dim=-1)
            sample_error = torch.mean(diff)  # ADE for each sample
            per_sample_errors.append(sample_error)
            total_error += diff.sum().item()
            count += diff.numel()
        if per_sample:
            return per_sample_errors.cpu().numpy()
        else:
            return total_error / count if count > 0 else 0.0
    else:
        if not isinstance(predictions, torch.Tensor):
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions)
            else:
                predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            else:
                targets = torch.tensor(targets)
        if per_sample:
            # Compute per-sample errors by averaging over time (dim=1)
            per_sample_errors = torch.mean(torch.norm(predictions - targets, dim=-1), dim=1)
            return per_sample_errors.cpu().numpy()
        else:
            return torch.mean(torch.norm(predictions - targets, dim=-1)).item()


def calculate_fde(predictions, targets,per_sample: bool = False):
    """
    Compute Final Displacement Error (FDE) as the mean Euclidean distance 
    between the predicted and target positions at the final time step.
    
    This function converts inputs to tensors if they are provided as lists or NumPy arrays.
    
    Args:
        predictions (torch.Tensor or list or np.ndarray): Predicted positions with shape (B, T, 2)
            or (B, T, N, 2). If a list, each element is assumed to be a tensor (or convertible)
            with shape (T, N_valid, 2).
        targets (torch.Tensor or list or np.ndarray): Ground-truth positions with the same shape as predictions.
        per_sample (bool): If True, returns a NumPy array of per-sample FDE values.
    
    Returns:
        float: The average FDE computed over all samples (and nodes, if applicable).
    """
    if isinstance(predictions, list):
        total_error = 0.0
        count = 0
        per_sample_errors = []
        for pred, targ in zip(predictions, targets):
            if not isinstance(pred, torch.Tensor):
                if isinstance(pred, np.ndarray):
                    pred = torch.from_numpy(pred)
                else:
                    pred = torch.tensor(pred)
            if not isinstance(targ, torch.Tensor):
                if isinstance(targ, np.ndarray):
                    targ = torch.from_numpy(targ)
                else:
                    targ = torch.tensor(targ)

            diff = torch.norm(pred[-1, :] - targ[-1, :], dim=-1) 
            per_sample_errors.append(torch.mean(diff))
            total_error += diff.sum().item()
            count += diff.numel()
           
        per_sample_errors = torch.stack(per_sample_errors)
        
        if per_sample:
            return per_sample_errors.cpu().numpy()
        else:
            return total_error / count if count > 0 else 0.0
    else:
        if not isinstance(predictions, torch.Tensor):
            if isinstance(predictions, np.ndarray):
                predictions = torch.from_numpy(predictions)
            else:
                predictions = torch.tensor(predictions)
        if not isinstance(targets, torch.Tensor):
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
            else:
                targets = torch.tensor(targets)
        if predictions.ndim == 3:
            # Compute per-sample errors (shape: [B])
            per_sample_errors = torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1)
            if per_sample:
                return per_sample_errors.cpu().numpy()
            else:
                return torch.mean(torch.norm(predictions[:, -1, :] - targets[:, -1, :], dim=-1)).item()
        elif predictions.ndim == 4:
            return torch.mean(torch.norm(predictions[:, -1, :, :] - targets[:, -1, :, :], dim=-1)).item()
        else:
            raise ValueError("Unsupported tensor shape for predictions/targets.")

class TrajectoryHandler:
    def __init__(self, max_objects=20, max_length=15, include_velocity=True,
                 normalize=False, mean=None, std=None, device='cuda'):
        """
        Trajectory handler compatible with model training/testing loops.
        
        Args:
            max_objects: Maximum number of objects to track
            max_length: Maximum length of trajectory history
            include_velocity: Whether to include velocity in outputs (default: True)
            normalize: Whether to normalize the outputs
            mean: Mean values for normalization [x, y, vx, vy]
            std: Standard deviation values for normalization [x, y, vx, vy]
            device: Device to store tensors on ('cuda' or 'cpu')
        """
        self.max_objects = max_objects
        self.max_length = max_length
        self.device = device
        self.include_velocity = include_velocity  # Should always be True for model compatibility
        self.normalize = normalize
        
        # Set normalization parameters
        if normalize:
            self.mean = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device) if mean is None else torch.tensor(mean, device=device)
            self.std = torch.tensor([1.0, 1.0, 1.0, 1.0], device=device) if std is None else torch.tensor(std, device=device)
        
        # Initialize trajectory storage (only raw positions)
        self.trajectories = torch.zeros((max_objects, max_length, 2), device=device)

        # valid trajectory points without padding 
        self.valid_len = torch.zeros(self.max_objects, dtype=torch.long, device=self.device)

        
        # Object tracking dictionaries
        self.obj_id_to_index = {}  # Maps object ID to index in trajectories tensor
        self.active_objects = set()  # Set of currently active objects
        self.available_index = [i for i in range(max_objects)]  # available indicies
        
    def _update_trajectory(self, obj_id, x, y):
        """
        Internal method to update a single object's trajectory.
        
        Args:
            obj_id: Object identifier
            x: X position
            y: Y position
        """
        # Add new object if not seen before
        if obj_id not in self.obj_id_to_index:
            if not self.available_index:
                return False
                    
            idx = self.available_index.pop()
            self.obj_id_to_index[obj_id] = idx

        # Get index and update trajectory
        idx = self.obj_id_to_index[obj_id]
        
        # Shift trajectory and add new position
        self.trajectories[idx, :-1] = self.trajectories[idx, 1:].clone()
        self.trajectories[idx, -1] = torch.tensor([x, y], device=self.device)
        self.valid_len[idx] = torch.clamp(self.valid_len[idx] + 1, max=self.max_length)
    
        return True
        
    def update(self, detections):
        """
        Update trajectories with new object detections.
        Automatically handles different input types based on instance.
        
        Args:
            detections: Can be one of:
                - List of (obj_id, x, y) tuples
                - ROS Detection3DArray message
                - PerceptionModel instance
        """
        # Handle different input types based on instance
        if hasattr(detections, 'detections'):
            # ROS Detection3DArray message
            detection_generator = self._parse_detection3d_array(detections)
        elif hasattr(detections, 'agent_vehicles'):
            # PerceptionModel instance
            detection_generator = self._parse_perception_model(detections)
        elif isinstance(detections, list):
            # List of tuples
            detection_generator = self._parse_tuple_list(detections)
        else:
            raise ValueError(
                "Unsupported detection format. Expected one of:\n"
                "- List of (obj_id, x, y) tuples\n"
                "- ROS Detection3DArray message (with .detections attribute)\n"
                "- PerceptionModel instance (with .agent_vehicles attribute)"
            )
        
        # Process all detections using common logic
        self._process_detections(detection_generator)
    
    def _process_detections(self, detection_generator):
        """
        Process detections from any source and update active objects.
        
        Args:
            detection_generator: Generator yielding (obj_id, x, y) tuples
        """
        prev_active = set(self.active_objects)
        updated_ids = set()
        
        for obj_id, x, y in detection_generator:
            if self._update_trajectory(obj_id, x, y):
                updated_ids.add(obj_id)
        
        dropped = prev_active - updated_ids
        for oid in dropped:
            self._free_id(oid)
        # Remove objects that weren't updated
        self.active_objects = updated_ids  # directly set
        log.debug(f"Active objects updated: {self.active_objects}")
    
    def _parse_tuple_list(self, detections):
        """Parse list of (obj_id, x, y) tuples."""
        if len(detections) > 0 and not isinstance(detections[0], tuple):
            raise ValueError("Each detection must be a tuple of (obj_id, x, y)")
        
        for detection in detections:
            if len(detection) != 3:
                raise ValueError("Each detection tuple must have exactly 3 elements: (obj_id, x, y)")
            yield detection
    
    def _parse_detection3d_array(self, detection_msg):
        """Parse ROS Detection3DArray message."""
        for detection in detection_msg.detections:
            if not detection.results:
                continue
            
            obj_id = int(detection.results[0].hypothesis.class_id)
            x = detection.bbox.center.position.x
            y = detection.bbox.center.position.y
            yield (obj_id, x, y)
    
    def _parse_perception_model(self, perception_model):
        """Parse PerceptionModel instance."""
        for agent in perception_model.agent_vehicles:
            yield (agent.agent_id, agent.x, agent.y)
            

    def _free_id(self, oid: int):
        """Remove ID and free its slot for reuse."""
        idx = self.obj_id_to_index.pop(oid, None)
        if idx is None:
            return
        self.index_to_obj_id.pop(idx, None)
        self.trajectories[idx].zero_()
        self.valid_len[idx] = 0
        self.available_index.append(idx)
    def __len__(self):
        """
        Returns the number of active objects.
        
        Returns:
            int: Number of active objects
        """
        return len(self.active_objects)
    
    def __getitem__(self, idx):
        """
        Retrieves the trajectory with velocity for a given object index.
        Format matches what's expected by the prediction model.
        
        Args:
            idx: Index of the object (not object ID)
            
        Returns:
            trajectory_tensor: Tensor with shape (max_length, 4) containing [x, y, vx, vy]
        """
        # Convert index to object ID
        if idx < 0 or idx >= len(self.active_objects):
            raise IndexError(f"Index {idx} out of range for {len(self.active_objects)} active objects")
        
        # Get object ID from index
        active_ids = list(self.active_objects)
        obj_id = active_ids[idx]
        
        # Get the object's trajectory
        idx = self.obj_id_to_index[obj_id]
        trajectory = self.trajectories[idx].clone()

        
        # Process observation - always include velocity for model compatibility
        # Compute velocities
        L = int(self.valid_len[idx]) # valid points in trajectory
        
        if L > 1:
            # Diff only valid tail; first valid step copies next frame's velocity
            diffs = trajectory[-L+1:] - trajectory[-L:-1]
            velocities = torch.zeros_like(trajectory)
            velocities[-L] = diffs[0]
            velocities[-L+1:] = diffs
        else:
            velocities = torch.zeros_like(trajectory)

        trajectory = torch.cat([trajectory, velocities], dim=1)
        
        # Normalize data if requested
        if self.normalize:
            trajectory = (trajectory - self.mean) / self.std
            
        return trajectory
    
    def get_active_object_ids(self):
        """
        Get list of active object IDs.
        
        Returns:
            List of currently active object IDs
        """
        return list(self.active_objects)
    
    def get_trajectory(self, obj_id):
        """
        Get trajectory for a specific object ID.
        
        Args:
            obj_id: Object ID
            
        Returns:
            trajectory_tensor: Tensor with shape (max_length, 4) containing [x, y, vx, vy]
            or None if object not found
        """
        if obj_id not in self.obj_id_to_index or obj_id not in self.active_objects:
            return None
            
        # Find the index of this object in the active objects list
        active_ids = list(self.active_objects)
        try:
            idx = active_ids.index(obj_id)
            return self[idx]  # Reuse __getitem__ to ensure consistent processing
        except ValueError:
            return None
    
    def get_batch(self):
        """
        Get a batch of all active trajectories for direct use with prediction models.
        
        Returns:
            Tensor batch with shape (num_objects, max_length, 4) containing [x, y, vx, vy]
        """
        if len(self) == 0:
            return torch.zeros((0, self.max_length, 4), device=self.device)
            
        # Create tensor to hold all trajectories
        batch = torch.zeros((len(self), self.max_length, 4), device=self.device)
        
        # Fill with trajectory data
        for i in range(len(self)):
            batch[i] = self[i]
            
        return batch
    
    def data_iter(self, batch_size=1):
        """
        Create an iterator that yields batches similar to a DataLoader.
        
        Args:
            batch_size: Number of objects per batch
            
        Returns:
            Generator yielding batches of shape (batch_size, max_length, 4)
        """
        indices = list(range(len(self)))
        
        # Process in batch_size chunks
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = torch.stack([self[j] for j in batch_indices])
            yield batch
