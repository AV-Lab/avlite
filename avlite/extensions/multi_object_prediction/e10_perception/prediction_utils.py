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


def visualize_occupancy_grid_animation(occupancy_grid, grid_bounds, mue=None, 
                                        title="Occupancy Grid Animation", figsize=(12, 8), 
                                        show_grid_cells=True, interval=500):
        """
        Interactive animation of occupancy grid with clean save functionality.
        
        Args:
            occupancy_grid: Shape (num_timesteps, grid_steps, grid_steps)
            grid_bounds: Dictionary with min_x, max_x, min_y, max_y
            mue: Mean vectors for overlay (optional)
            title: Animation title
            figsize: Figure size
            show_grid_cells: Whether to show grid cell boundaries
            interval: Animation interval in milliseconds
        """
        if isinstance(occupancy_grid, torch.Tensor):
            grid_data = occupancy_grid.cpu().numpy()
        else:
            grid_data = occupancy_grid
        
        if mue is not None and isinstance(mue, torch.Tensor):
            means_data = mue.cpu().numpy()
        else:
            means_data = mue
        
        num_timesteps, grid_steps, _ = grid_data.shape
        
        # Create figure with smaller control area
        fig = plt.figure(figsize=figsize)
        
        # Adjusted layout - main plot takes most space, small control area at bottom
        ax_main = plt.subplot2grid((20, 20), (0, 0), colspan=20, rowspan=17)
        
        # Smaller buttons at the bottom
        ax_play = plt.subplot2grid((20, 20), (18, 2), colspan=3, rowspan=1)
        ax_pause = plt.subplot2grid((20, 20), (18, 6), colspan=3, rowspan=1)
        ax_save = plt.subplot2grid((20, 20), (18, 10), colspan=3, rowspan=1)
        ax_slider = plt.subplot2grid((20, 20), (18, 15), colspan=4, rowspan=1)
        
        # Initialize the main plot
        im = ax_main.imshow(grid_data[0], 
                        origin='lower',
                        extent=[grid_bounds['min_x'], grid_bounds['max_x'], 
                                grid_bounds['min_y'], grid_bounds['max_y']],
                        cmap='hot',
                        vmin=0, vmax=1,
                        aspect='equal')
        
        # Add grid lines if requested
        if show_grid_cells:
            x_edges = np.linspace(grid_bounds['min_x'], grid_bounds['max_x'], grid_steps + 1)
            y_edges = np.linspace(grid_bounds['min_y'], grid_bounds['max_y'], grid_steps + 1)
            
            for x in x_edges:
                ax_main.axvline(x=x, color='white', linewidth=0.3, alpha=0.5)
            for y in y_edges:
                ax_main.axhline(y=y, color='white', linewidth=0.3, alpha=0.5)
        
        # Initialize scatter plots for means (if provided)
        scatter_plots = []
        if means_data is not None:
            colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            
            num_objects, _, num_components, _ = means_data.shape
            
            for obj_idx in range(num_objects):
                color = colors[obj_idx % len(colors)]
                marker = markers[obj_idx % len(markers)]
                
                obj_means = means_data[obj_idx, 0, :, :]
                scatter = ax_main.scatter(obj_means[:, 0], obj_means[:, 1], 
                                        c=color, s=60, marker=marker, 
                                        label=f'Object {obj_idx+1}', 
                                        edgecolors='white', linewidth=1, alpha=0.9, zorder=5)
                scatter_plots.append(scatter)
        
        # Set up the plot
        plt.colorbar(im, ax=ax_main, label='Occupancy Probability', shrink=0.8)
        ax_main.set_xlabel('X Position')
        ax_main.set_ylabel('Y Position')
        title_text = ax_main.set_title(f'{title} - Timestep 0/{num_timesteps-1}')
        
        if means_data is not None:
            ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Animation state
        class AnimationState:
            def __init__(self):
                self.current_frame = 0
                self.is_playing = False
                self.anim = None
        
        state = AnimationState()
        
        # Animation function for display
        def animate_display(frame):
            state.current_frame = frame
            
            im.set_array(grid_data[frame])
            
            if means_data is not None:
                for obj_idx, scatter in enumerate(scatter_plots):
                    obj_means = means_data[obj_idx, frame, :, :]
                    scatter.set_offsets(obj_means)
            
            title_text.set_text(f'{title} - Timestep {frame}/{num_timesteps-1}')
            
            if hasattr(animate_display, 'slider'):
                animate_display.slider.set_val(frame)
            
            return [im, title_text] + scatter_plots
        
        # Clean animation function for saving (grid only)
        def create_clean_animation():
            """Create a clean figure with just the grid for saving."""
            # Create a clean figure for saving
            save_fig, save_ax = plt.subplots(figsize=(8, 8))
            
            # Initialize clean plot
            save_im = save_ax.imshow(grid_data[0], 
                                origin='lower',
                                extent=[grid_bounds['min_x'], grid_bounds['max_x'], 
                                        grid_bounds['min_y'], grid_bounds['max_y']],
                                cmap='hot',
                                vmin=0, vmax=1,
                                aspect='equal')
            
            # Add grid lines if requested (thinner for cleaner look)
            if show_grid_cells:
                x_edges = np.linspace(grid_bounds['min_x'], grid_bounds['max_x'], grid_steps + 1)
                y_edges = np.linspace(grid_bounds['min_y'], grid_bounds['max_y'], grid_steps + 1)
                
                for x in x_edges:
                    save_ax.axvline(x=x, color='white', linewidth=0.2, alpha=0.3)
                for y in y_edges:
                    save_ax.axhline(y=y, color='white', linewidth=0.2, alpha=0.3)
            
            # Initialize means for clean animation
            save_scatter_plots = []
            if means_data is not None:
                colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
                markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
                
                for obj_idx in range(num_objects):
                    color = colors[obj_idx % len(colors)]
                    marker = markers[obj_idx % len(markers)]
                    
                    obj_means = means_data[obj_idx, 0, :, :]
                    scatter = save_ax.scatter(obj_means[:, 0], obj_means[:, 1], 
                                            c=color, s=80, marker=marker, 
                                            edgecolors='white', linewidth=2, alpha=0.9, zorder=5)
                    save_scatter_plots.append(scatter)
            
            plt.colorbar(save_im, ax=save_ax, label='Occupancy Probability')
            save_ax.set_xlabel('X Position', fontsize=12)
            save_ax.set_ylabel('Y Position', fontsize=12)
            save_title = save_ax.set_title(f'{title} - Timestep 0', fontsize=14, pad=20)
            
            plt.tight_layout()
            
            # Animation function for clean save
            def animate_clean(frame):
                save_im.set_array(grid_data[frame])
                save_title.set_text(f'{title} - Timestep {frame}')
                
                if means_data is not None:
                    for obj_idx, scatter in enumerate(save_scatter_plots):
                        obj_means = means_data[obj_idx, frame, :, :]
                        scatter.set_offsets(obj_means)
                
                return [save_im, save_title] + save_scatter_plots
            
            return save_fig, animate_clean
        
        # Control functions
        def start_animation():
            if state.anim is None or not state.is_playing:
                state.anim = animation.FuncAnimation(fig, animate_display, frames=num_timesteps,
                                                interval=interval, blit=False, repeat=True)
                state.is_playing = True
                plt.draw()
        
        def stop_animation():
            if state.anim is not None:
                state.anim.pause()
                state.is_playing = False
        
        def save_animation():
            print("Creating clean animation for saving...")
            
            # Create clean animation
            save_fig, animate_clean = create_clean_animation()
            clean_anim = animation.FuncAnimation(save_fig, animate_clean, frames=num_timesteps,
                                            interval=interval, blit=False, repeat=False)
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"occupancy_grid_{timestamp}"
            
            saved_files = []
            
            # Save as GIF
            gif_path = f"{base_name}.gif"
            try:
                print(f"Saving GIF: {gif_path}")
                writer = animation.PillowWriter(fps=max(1, 1000//interval), 
                                            metadata=dict(artist='OccupancyGrid'))
                clean_anim.save(gif_path, writer=writer, dpi=150)
                saved_files.append(gif_path)
                print(f"✅ GIF saved: {gif_path}")
            except Exception as e:
                print(f"❌ GIF save failed: {e}")
            
            # Save as MP4 (if available)
            mp4_path = f"{base_name}.mp4"
            try:
                print(f"Saving MP4: {mp4_path}")
                writer = animation.FFMpegWriter(fps=max(1, 1000//interval),
                                            metadata=dict(artist='OccupancyGrid'),
                                            bitrate=2000, extra_args=['-vcodec', 'libx264'])
                clean_anim.save(mp4_path, writer=writer, dpi=150)
                saved_files.append(mp4_path)
                print(f"✅ MP4 saved: {mp4_path}")
            except Exception as e:
                print(f"⚠️ MP4 save failed (ffmpeg may not be available): {e}")
            
            # Close the save figure to free memory
            plt.close(save_fig)
            
            if saved_files:
                print(f"✅ Clean grid animation(s) saved: {', '.join(saved_files)}")
            else:
                print("❌ No files were saved")
        
        # Manual timestep control
        def update_timestep(val):
            frame = int(val)
            if frame != state.current_frame:
                animate_display(frame)
                plt.draw()
        
        # Create smaller controls
        slider = Slider(ax_slider, 'Step', 0, num_timesteps-1, 
                    valinit=0, valfmt='%d')
        slider.on_changed(update_timestep)
        animate_display.slider = slider
        
        # Create smaller buttons
        btn_play = Button(ax_play, '▶ Play', color='lightgreen', hovercolor='green')
        btn_pause = Button(ax_pause, '⏸ Pause', color='lightcoral', hovercolor='red')  
        btn_save = Button(ax_save, '💾 Save', color='lightblue', hovercolor='blue')
        
        # Make button text smaller
        btn_play.label.set_fontsize(8)
        btn_pause.label.set_fontsize(8)
        btn_save.label.set_fontsize(8)
        
        # Button callbacks
        btn_play.on_clicked(lambda x: start_animation())
        btn_pause.on_clicked(lambda x: stop_animation())
        btn_save.on_clicked(lambda x: save_animation())
        
        # Keyboard shortcuts
        def on_key_press(event):
            if event.key == ' ':
                if state.is_playing:
                    stop_animation()
                else:
                    start_animation()
            elif event.key == 's':
                save_animation()
            elif event.key == 'left':
                new_frame = max(0, state.current_frame - 1)
                slider.set_val(new_frame)
            elif event.key == 'right':
                new_frame = min(num_timesteps - 1, state.current_frame + 1)
                slider.set_val(new_frame)
            elif event.key == 'escape':
                stop_animation()
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Add compact instructions
        instruction_text = "Controls: Space=Play/Pause | S=Save | ←/→=Step | Click buttons"
        fig.text(0.5, 0.02, instruction_text, ha='center', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)
        
        # Start with first frame
        animate_display(0)
        
        plt.show()
        
        return state



def visualize_occupancy_grid_tkinter(parent_window, occupancy_grid, grid_bounds, mue=None, 
                                   title="Occupancy Grid Animation", figsize=(10, 8), 
                                   show_grid_cells=True, interval=500):
    """
    Tkinter-integrated occupancy grid animation.
    
    Args:
        parent_window: Parent Tkinter window (can be None for standalone)
        occupancy_grid: Shape (num_timesteps, grid_steps, grid_steps)
        grid_bounds: Dictionary with min_x, max_x, min_y, max_y
        mue: Mean vectors for overlay (optional)
        title: Animation title
        figsize: Figure size
        show_grid_cells: Whether to show grid cell boundaries
        interval: Animation interval in milliseconds
    """
    if isinstance(occupancy_grid, torch.Tensor):
        grid_data = occupancy_grid.cpu().numpy()
    else:
        grid_data = occupancy_grid
    
    if mue is not None and isinstance(mue, torch.Tensor):
        means_data = mue.cpu().numpy()
    else:
        means_data = mue
    
    num_timesteps, grid_steps, _ = grid_data.shape
    
    # Create new Tkinter window
    if parent_window is None:
        window = tk.Tk()
    else:
        window = tk.Toplevel(parent_window)
    
    window.title(title)
    window.geometry("1000x700")
    
    # Create matplotlib figure (no plt.figure()!)
    fig = Figure(figsize=figsize, dpi=100)
    ax_main = fig.add_subplot(111)
    
    # Initialize the main plot
    im = ax_main.imshow(grid_data[0], 
                    origin='lower',
                    extent=[grid_bounds['min_x'], grid_bounds['max_x'], 
                            grid_bounds['min_y'], grid_bounds['max_y']],
                    cmap='hot',
                    vmin=0, vmax=1,
                    aspect='equal')
    
    # Add grid lines if requested
    if show_grid_cells:
        x_edges = np.linspace(grid_bounds['min_x'], grid_bounds['max_x'], grid_steps + 1)
        y_edges = np.linspace(grid_bounds['min_y'], grid_bounds['max_y'], grid_steps + 1)
        
        for x in x_edges:
            ax_main.axvline(x=x, color='white', linewidth=0.3, alpha=0.5)
        for y in y_edges:
            ax_main.axhline(y=y, color='white', linewidth=0.3, alpha=0.5)
    
    # Initialize scatter plots for means (if provided)
    scatter_plots = []
    if means_data is not None:
        colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        
        num_objects, _, num_components, _ = means_data.shape
        
        for obj_idx in range(num_objects):
            color = colors[obj_idx % len(colors)]
            marker = markers[obj_idx % len(markers)]
            
            obj_means = means_data[obj_idx, 0, :, :]
            scatter = ax_main.scatter(obj_means[:, 0], obj_means[:, 1], 
                                    c=color, s=60, marker=marker, 
                                    label=f'Object {obj_idx+1}', 
                                    edgecolors='white', linewidth=1, alpha=0.9, zorder=5)
            scatter_plots.append(scatter)
    
    # Set up the plot
    fig.colorbar(im, ax=ax_main, label='Occupancy Probability')
    ax_main.set_xlabel('X Position')
    ax_main.set_ylabel('Y Position')
    title_text = ax_main.set_title(f'{title} - Timestep 0/{num_timesteps-1}')
    
    if means_data is not None:
        ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    fig.tight_layout()
    
    # Embed matplotlib in Tkinter
    canvas = FigureCanvasTkAgg(fig, window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Animation state
    class AnimationState:
        def __init__(self):
            self.current_frame = 0
            self.is_playing = False
            self.anim = None
    
    state = AnimationState()
    
    # Animation function
    def animate_display(frame):
        state.current_frame = frame
        
        im.set_array(grid_data[frame])
        
        if means_data is not None:
            for obj_idx, scatter in enumerate(scatter_plots):
                obj_means = means_data[obj_idx, frame, :, :]
                scatter.set_offsets(obj_means)
        
        title_text.set_text(f'{title} - Timestep {frame}/{num_timesteps-1}')
        
        # Update slider without triggering callback
        if hasattr(animate_display, 'slider_var'):
            animate_display.slider_var.set(frame)
        
        canvas.draw_idle()  # More efficient than canvas.draw()
        return [im, title_text] + scatter_plots
    
    # Control functions
    def start_animation():
        if state.anim is None or not state.is_playing:
            state.anim = animation.FuncAnimation(fig, animate_display, frames=num_timesteps,
                                            interval=interval, blit=False, repeat=True)
            state.is_playing = True
            btn_play.config(state='disabled')
            btn_pause.config(state='normal')
    
    def stop_animation():
        if state.anim is not None:
            state.anim.pause()
            state.is_playing = False
            btn_play.config(state='normal')
            btn_pause.config(state='disabled')
    
    def save_animation():
        print("Creating clean animation for saving...")
        
        # Create clean figure for saving
        save_fig = Figure(figsize=(8, 8), dpi=150)
        save_ax = save_fig.add_subplot(111)
        
        # Initialize clean plot
        save_im = save_ax.imshow(grid_data[0], 
                            origin='lower',
                            extent=[grid_bounds['min_x'], grid_bounds['max_x'], 
                                    grid_bounds['min_y'], grid_bounds['max_y']],
                            cmap='hot',
                            vmin=0, vmax=1,
                            aspect='equal')
        
        # Add grid lines if requested
        if show_grid_cells:
            x_edges = np.linspace(grid_bounds['min_x'], grid_bounds['max_x'], grid_steps + 1)
            y_edges = np.linspace(grid_bounds['min_y'], grid_bounds['max_y'], grid_steps + 1)
            
            for x in x_edges:
                save_ax.axvline(x=x, color='white', linewidth=0.2, alpha=0.3)
            for y in y_edges:
                save_ax.axhline(y=y, color='white', linewidth=0.2, alpha=0.3)
        
        # Initialize means for clean animation
        save_scatter_plots = []
        if means_data is not None:
            colors = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown']
            markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
            
            for obj_idx in range(num_objects):
                color = colors[obj_idx % len(colors)]
                marker = markers[obj_idx % len(markers)]
                
                obj_means = means_data[obj_idx, 0, :, :]
                scatter = save_ax.scatter(obj_means[:, 0], obj_means[:, 1], 
                                        c=color, s=80, marker=marker, 
                                        edgecolors='white', linewidth=2, alpha=0.9, zorder=5)
                save_scatter_plots.append(scatter)
        
        save_fig.colorbar(save_im, ax=save_ax, label='Occupancy Probability')
        save_ax.set_xlabel('X Position', fontsize=12)
        save_ax.set_ylabel('Y Position', fontsize=12)
        save_title = save_ax.set_title(f'{title} - Timestep 0', fontsize=14, pad=20)
        
        save_fig.tight_layout()
        
        # Animation function for clean save
        def animate_clean(frame):
            save_im.set_array(grid_data[frame])
            save_title.set_text(f'{title} - Timestep {frame}')
            
            if means_data is not None:
                for obj_idx, scatter in enumerate(save_scatter_plots):
                    obj_means = means_data[obj_idx, frame, :, :]
                    scatter.set_offsets(obj_means)
            
            return [save_im, save_title] + save_scatter_plots
        
        # Create and save animation
        clean_anim = animation.FuncAnimation(save_fig, animate_clean, frames=num_timesteps,
                                        interval=interval, blit=False, repeat=False)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"occupancy_grid_{timestamp}"
        
        saved_files = []
        
        # Update status
        status_label.config(text="Saving animation...")
        window.update()
        
        # Save as GIF
        gif_path = f"{base_name}.gif"
        try:
            print(f"Saving GIF: {gif_path}")
            writer = animation.PillowWriter(fps=max(1, 1000//interval), 
                                        metadata=dict(artist='OccupancyGrid'))
            clean_anim.save(gif_path, writer=writer, dpi=150)
            saved_files.append(gif_path)
            print(f"✅ GIF saved: {gif_path}")
        except Exception as e:
            print(f"❌ GIF save failed: {e}")
        
        # Save as MP4 (if available)
        mp4_path = f"{base_name}.mp4"
        try:
            print(f"Saving MP4: {mp4_path}")
            writer = animation.FFMpegWriter(fps=max(1, 1000//interval),
                                        metadata=dict(artist='OccupancyGrid'),
                                        bitrate=2000, extra_args=['-vcodec', 'libx264'])
            clean_anim.save(mp4_path, writer=writer, dpi=150)
            saved_files.append(mp4_path)
            print(f"✅ MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"⚠️ MP4 save failed (ffmpeg may not be available): {e}")
        
        if saved_files:
            status_label.config(text=f"Saved: {', '.join(saved_files)}")
            print(f"✅ Clean grid animation(s) saved: {', '.join(saved_files)}")
        else:
            status_label.config(text="Save failed!")
            print("❌ No files were saved")
    
    # Manual timestep control
    def update_timestep(*args):
        frame = int(slider_var.get())
        if frame != state.current_frame:
            animate_display(frame)
    
    # Create control frame
    control_frame = tk.Frame(window)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
    
    # Create controls row 1
    controls1 = tk.Frame(control_frame)
    controls1.pack(fill=tk.X, pady=5)
    
    # Playback controls
    btn_play = tk.Button(controls1, text="▶ Play", command=start_animation, 
                        bg='lightgreen', width=8)
    btn_play.pack(side=tk.LEFT, padx=5)
    
    btn_pause = tk.Button(controls1, text="⏸ Pause", command=stop_animation, 
                         bg='lightcoral', state='disabled', width=8)
    btn_pause.pack(side=tk.LEFT, padx=5)
    
    btn_save = tk.Button(controls1, text="💾 Save", command=save_animation, 
                        bg='lightblue', width=8)
    btn_save.pack(side=tk.LEFT, padx=5)
    
    # Step controls
    def step_backward():
        new_frame = max(0, state.current_frame - 1)
        slider_var.set(new_frame)
    
    def step_forward():
        new_frame = min(num_timesteps - 1, state.current_frame + 1)
        slider_var.set(new_frame)
    
    btn_back = tk.Button(controls1, text="◀", command=step_backward, width=3)
    btn_back.pack(side=tk.LEFT, padx=5)
    
    btn_forward = tk.Button(controls1, text="▶", command=step_forward, width=3)
    btn_forward.pack(side=tk.LEFT, padx=5)
    
    # Status label
    status_label = tk.Label(controls1, text="Ready", relief=tk.SUNKEN, width=20)
    status_label.pack(side=tk.RIGHT, padx=5)
    
    # Create controls row 2 - Slider
    controls2 = tk.Frame(control_frame)
    controls2.pack(fill=tk.X, pady=5)
    
    tk.Label(controls2, text="Timestep:").pack(side=tk.LEFT, padx=5)
    
    slider_var = tk.IntVar()
    slider_var.set(0)
    slider_var.trace('w', update_timestep)
    animate_display.slider_var = slider_var
    
    slider = tk.Scale(controls2, from_=0, to=num_timesteps-1, 
                     orient=tk.HORIZONTAL, variable=slider_var,
                     length=400)
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
    
    # Add instructions
    instructions = tk.Label(control_frame, 
                           text="Controls: Play/Pause buttons | ◀/▶ for single steps | Drag slider | Space=Play/Pause | S=Save",
                           font=('Arial', 9), fg='gray')
    instructions.pack(pady=5)
    
    # Keyboard shortcuts
    def on_key_press(event):
        if event.keysym == 'space':
            if state.is_playing:
                stop_animation()
            else:
                start_animation()
        elif event.keysym == 's':
            save_animation()
        elif event.keysym == 'Left':
            step_backward()
        elif event.keysym == 'Right':
            step_forward()
        elif event.keysym == 'Escape':
            stop_animation()
    
    # Bind keyboard events to window
    window.bind('<KeyPress>', on_key_press)
    window.focus_set()  # Make sure window can receive key events
    
    # Start with first frame
    animate_display(0)
    
    # Handle window closing
    def on_closing():
        if state.anim is not None:
            state.anim.pause()
        window.destroy()
    
    window.protocol("WM_DELETE_WINDOW", on_closing)
    
    return window, state
def visualize_multiple_timesteps(self,occupancy_grid, grid_bounds, timesteps=None, 
                                cols=3, figsize=(15, 10)):
        """
        Visualize multiple timesteps in a grid layout.
        """
        if isinstance(occupancy_grid, torch.Tensor):
            grid_data = occupancy_grid.cpu().numpy()
        else:
            grid_data = occupancy_grid
        
        num_timesteps = grid_data.shape[0]
        
        if timesteps is None:
            # Show evenly spaced timesteps
            timesteps = np.linspace(0, num_timesteps-1, min(9, num_timesteps)).astype(int)
        
        rows = (len(timesteps) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        if len(timesteps) == 1:
            axes = axes.reshape(1, 1)
        
        for idx, t in enumerate(timesteps):
            row = idx // cols
            col = idx % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            im = ax.imshow(grid_data[t], 
                        origin='lower',
                        extent=[grid_bounds['min_x'], grid_bounds['max_x'], 
                                grid_bounds['min_y'], grid_bounds['max_y']],
                        cmap='hot',
                        vmin=0, vmax=1,
                        aspect='equal')
            
            ax.set_title(f'Timestep {t}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        total_plots = rows * cols
        for idx in range(len(timesteps), total_plots):
            row = idx // cols
            col = idx % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        # Add colorbar
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axes, label='Occupancy Probability', shrink=0.8)
        
        plt.show()
 
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
        
        # Object tracking dictionaries
        self.obj_id_to_index = {}  # Maps object ID to index in trajectories tensor
        self.index_to_obj_id = {}  # Maps index to object ID
        self.active_objects = set()  # Set of currently active objects
        self.next_index = 0  # Next available index
        
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
            if self.next_index >= self.max_objects:
                return False  # Skip if at max capacity
                
            self.obj_id_to_index[obj_id] = self.next_index
            self.index_to_obj_id[self.next_index] = obj_id
            self.next_index += 1
        
        # Get index and update trajectory
        idx = self.obj_id_to_index[obj_id]
        
        # Shift trajectory and add new position
        self.trajectories[idx, :-1] = self.trajectories[idx, 1:].clone()
        self.trajectories[idx, -1] = torch.tensor([x, y], device=self.device)
        
        # Mark as active
        self.active_objects.add(obj_id)
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
        updated_ids = set()
        
        for obj_id, x, y in detection_generator:
            if self._update_trajectory(obj_id, x, y):
                updated_ids.add(obj_id)
        
        # Remove objects that weren't updated from active set
        self.active_objects = self.active_objects.intersection(updated_ids)
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
        if trajectory.size(0) > 1:
            velocities = trajectory[1:] - trajectory[:-1]
            # For first point, repeat first velocity
            velocities = torch.cat([velocities[0:1], velocities], dim=0)
            trajectory = torch.cat([trajectory, velocities], dim=1)
        else:
            # Handle single point case
            trajectory = torch.cat([trajectory, torch.zeros_like(trajectory)], dim=1)
        
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
