import torch
from utils.loading_utils import load_model, get_device
from utils.event_readers import EventDataset 
from utils.event_tensor_utils import events_to_voxel_grid_pytorch
from os.path import join, basename
import numpy as np
import json
import argparse
import shutil
import os
from depth_prediction import DepthEstimator
from options.inference_options import set_depth_inference_options
import random

class SimpleNamespace(object):
    def init(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

# Add filtering code here
class AdaptiveTemporalDensityFilter: # Renamed class for clarity
    #def __init__(self, resolution, adaptation_rate=0.06, decay_rate=0.05, density_threshold=1.0): # Added density_threshold
    def __init__(self, resolution, adaptation_rate=0.06, decay_rate=0.025, density_threshold=1.5): # Added density_threshold
        self.resolution = resolution
        self.adaptation_rate = adaptation_rate
        self.decay_rate = decay_rate
        self.density_map = np.zeros(resolution, dtype=np.float32) # Renamed to density_map
        self.density_threshold = density_threshold # New parameter

    def apply(self, events):
        filtered_events = []
        num_events_before = len(events)

        for event in events:
            x, y = int(event[1]), int(event[2])
            valid_coords = (0 <= x < self.resolution[0] and 0 <= y < self.resolution[1])

            if valid_coords:
                current_density = self.density_map[x, y] # Renamed to current_density
                self.density_map[x, y] += self.adaptation_rate # Still adapt density map
                if current_density <= self.density_threshold: # Filtering based on density threshold
                    filtered_events.append(event)
                self.density_map[x, y] *= (1 - self.decay_rate)

            else:
                print("Invalid coordinates!", x, y, self.resolution)

        #print("After applying: min density =", np.min(self.density_map), " max density =", np.max(self.density_map)) # Renamed to density
        #print("Reduction: %d events" % (num_events_before - len(filtered_events)))
        return np.array(filtered_events)

def density_filter(events, resolution, min_density=0, max_density=5): # density_filter function
    width, height = resolution
    grid = np.zeros((width, height), dtype=np.int32)
    #print('In density: ', events.shape)
    for event in events:
        x, y = int(event[1]), int(event[2])
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] += 1
    #print(max(grid.flatten()))
    return np.array([event for event in events if min_density <= grid[int(event[1]), int(event[2])] <= max_density])

def density_filter_new(events, resolution, min_density=0, start_downsample_density=40, max_density=872, max_downsample_rate=0.95):
    width, height = resolution
    grid = np.zeros((width, height), dtype=np.int32)

    # Calculate event densities
    for event in events:
        x, y = int(event[1]), int(event[2])
        if 0 <= x < width and 0 <= y < height:
            grid[x, y] += 1

    filtered_events = []
    for event in events:
        x, y = int(event[1]), int(event[2])
        if not (0 <= x < width and 0 <= y < height):
            continue  # Skip events outside valid bounds

        density = grid[x, y]

        if density < start_downsample_density:
            # Keep the event
            filtered_events.append(event)
        else:
            # Calculate a downsample rate, starting at 0.0 and approaching max_downsample_rate
            # as density increases, reaching max at max_density, to ensure high numbers
            downsample_rate = min(max_downsample_rate * (density - start_downsample_density) / (max_density - start_downsample_density), max_downsample_rate)

            # Invert for actual application - this is now the keep rate
            keep_rate = 1 - downsample_rate

            # Downsample the events based on downsample_rate
            if random.random() < keep_rate: # Probability of keeping the event
                filtered_events.append(event)

    return np.array(filtered_events)

def save_events(events, output_file): # save_events function
    with open(output_file, "w") as f:
        for event in events:
            f.write(f"{event[0]},{event[1]},{event[2]},{event[3]}\n")

def load_event_data(npy_folder): # load_event_data function
    event_files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
    all_events = []
    for file in event_files:
        file_path = os.path.join(npy_folder, file)
        events = np.load(file_path)  # Load events (timestamp, x, y, polarity)
        
        for event in events:
            timestamp, x, y, polarity = event
            # Create dv.Event and add to EventStore
            all_events.append(event)
    
    return np.array(all_events)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('-i', '--input_folder', default=None, type=str,
                        help="name of the folder containing the voxel grids")
    parser.add_argument('--start_time', default=0.0, type=float)
    parser.add_argument('--stop_time', default=0.0, type=float)
    #parser.add_argument('--event_filter', choices=['adaptiveThreshold', 'density'], type=str, default=None,
    #                    help="Choose event filtering method: 'adaptiveThreshold' or 'density'. Default is None.")
    #parser.set_defaults(event_filter=None)

    set_depth_inference_options(parser)

    args = parser.parse_args()

    print_every_n = 50

    # Load model to device
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)
    model = model.to(device)
    model.eval()

    base_folder = os.path.dirname(args.input_folder)
    event_folder = os.path.basename(args.input_folder)

    # hack to get the image size: create a dummy dataset,
    # grab the first data item and read the required info
    dummy_dataset = EventDataset(base_folder,
                                     event_folder,
                                     args.start_time,
                                     args.stop_time,
                                     transform=None)
    
    #data = dummy_dataset[0]
    #print('SHAPE DATA ', data['events'].shape)
    height, width = 260, 346
    #print('height ', height, 'width ', width)
    height = height - args.low_border_crop
    
    estimator = DepthEstimator(model, height, width, model.num_bins, args)


    output_dir = args.output_folder
    dataset_name = args.dataset_name
    print('Processing {}'.format(dataset_name), end=': ')

    N = len(dummy_dataset)

    if output_dir is not None:
        shutil.copyfile(join(args.input_folder, 'timestamps.txt'),
                        join(output_dir, dataset_name, 'timestamps.txt'))
        shutil.copyfile(join(args.input_folder, 'boundary_timestamps.txt'),
                        join(output_dir, dataset_name, 'boundary_timestamps.txt'))

    idx = 0
    filter = AdaptiveTemporalDensityFilter(resolution=(346, 260), adaptation_rate=0.06, decay_rate=0.025, density_threshold=1.5)
    event_count = 0
    filter_count = 0
    while idx < N:
        if idx % print_every_n == 0:
            print('{} / {}'.format(idx, N))
            print((1 - (filter_count+1)/(event_count+1))*100, '% events filtered')

        events_np = np.load(join(base_folder, event_folder , 'events_{:010d}.npy'.format(idx))) 
        event_count += events_np.shape[0]
        #events_np = density_filter(events_np, resolution=(width,height))
        events_np = filter.apply(events_np)
        filter_count += events_np.shape[0]
        events = torch.from_numpy(events_np) 
        events=events.to('cpu') 
        event_tensor = events_to_voxel_grid_pytorch(events.cpu().numpy(), 5, width, height, device)
        #print('events: ', events.shape,'tensor: ', event_tensor.shape)
        #event_tensor = data['events'][:height,:].float() 

        estimator.update_reconstruction(event_tensor, idx)
        idx += 1

print(f'{event_count - filter_count} events out of {event_count} were removed')