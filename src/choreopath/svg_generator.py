import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, Tuple
from .tracking_data import TrackingData
import mediapipe as mp

class SVGGenerator:
    """Generates SVG trajectories from body tracking data."""
    
    def __init__(self, width: int = 1280, height: int = 720, show_legend: bool = True):
        self.width = width
        self.height = height
        self.show_legend = show_legend
        
        # Color scheme for different body regions
        self.body_colors = {
            'face': '#f87171',
            'left_arm': '#fb923c',
            'right_arm': '#facc15',
            'hips': '#71717a',
            'left_leg': '#06b6d4',
            'right_leg': '#3b82f6'
        }

        # Hierarchical body structure mapping
        self.body_hierarchy = {
            # Face landmarks
            0:  ["body", "face"],
            1:  ["body", "face", "left eye"],
            2:  ["body", "face", "left eye"],
            3:  ["body", "face", "left eye"],
            4:  ["body", "face", "right eye"],
            5:  ["body", "face", "right eye"],
            6:  ["body", "face", "right eye"],
            7:  ["body", "face", "ears"],
            8:  ["body", "face", "ears"],
            9:  ["body", "face", "mouth"],
            10: ["body", "face", "mouth"],
            
            # Arm landmarks
            11: ["body", "left arm"],
            12: ["body", "right arm"],
            13: ["body", "left arm"],
            14: ["body", "right arm"],
            15: ["body", "left arm"],
            16: ["body", "right arm"],
            17: ["body", "left arm", "left hand"],
            18: ["body", "right arm", "right hand"],
            19: ["body", "left arm", "left hand"],
            20: ["body", "right arm", "right hand"],
            21: ["body", "left arm", "left hand"],
            22: ["body", "right arm", "right hand"],
            
            # Torso landmarks
            23: ["body", "hips"],
            24: ["body", "hips"],
            
            # Leg landmarks
            25: ["body", "left leg"],
            26: ["body", "right leg"],
            27: ["body", "left leg"],
            28: ["body", "right leg"],
            29: ["body", "left leg"],
            30: ["body", "right leg"],
            31: ["body", "left leg"],
            32: ["body", "right leg"]
        }
    
    def get_landmark_color(self, landmark_id: int) -> str:
        """Get color for a specific landmark based on body region."""
        if landmark_id <= 10:
            return self.body_colors['face']
        elif landmark_id in [11, 13, 15, 17, 19, 21]:
            return self.body_colors['left_arm']
        elif landmark_id in [12, 14, 16, 18, 20, 22]:
            return self.body_colors['right_arm']
        elif landmark_id in [23, 24]:
            return self.body_colors['hips']
        elif landmark_id in [25, 27, 29, 31]:
            return self.body_colors['left_leg']
        elif landmark_id in [26, 28, 30, 32]:
            return self.body_colors['right_leg']
        else:
            return '#888888'  # Gray fallback
    
    def normalize_to_svg_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert normalized coordinates (0-1) to SVG coordinates.
        """
        svg_x = x * self.width
        svg_y = y * self.height
        return svg_x, svg_y
    
    def create_layers(self, svg_root: ET.Element) -> Dict[str, ET.Element]:
        """
        Create nested layers elements based on body hierarchy.
        
        Returns a dictionary mapping group paths to their corresponding XML elements.
        """
        groups = {}
        
        # Collect all unique group paths from the hierarchy
        all_paths = set()
        for landmark_id, path in self.body_hierarchy.items():
            # Add all intermediate paths (e.g., for "Body/Face/Eyes/Left Eye", 
            # we want "Body", "Body/Face", "Body/Face/Eyes", "Body/Face/Eyes/Left Eye")
            for i in range(1, len(path) + 1):
                group_path = "/".join(path[:i])
                all_paths.add(group_path)

        # Sort paths by depth to ensure parent groups are created before children
        sorted_paths = sorted(all_paths, key=lambda x: x.count('/'))

        # Create group elements
        for path in sorted_paths:
            path_parts = path.split('/')
            group_name = path_parts[-1]  # Last part is the group name
            
            if len(path_parts) == 1:
                # Root level group - add directly to SVG root
                group_elem = ET.SubElement(svg_root, 'g')
                group_elem.set('inkscape:label', group_name)
                group_elem.set('id', group_name.lower().replace(' ', '_'))
                groups[path] = group_elem
            else:
                # Child group - add to parent group
                parent_path = "/".join(path_parts[:-1])
                parent_group = groups[parent_path]

                group_elem = ET.SubElement(parent_group, 'g')
                group_elem.set('inkscape:label', group_name)
                group_elem.set('id', path.lower().replace(' ', '_').replace('/', '_'))
                groups[path] = group_elem
        
        return groups
    
    def generate(self, tracking_data: TrackingData) -> ET.ElementTree:
        """
        Generate SVG trajectory visualization.
        """

        # Create SVG root
        svg_root = ET.Element('svg')
        svg_root.set('xmlns', 'http://www.w3.org/2000/svg')
        svg_root.set('width', str(self.width))
        svg_root.set('height', str(self.height))
        svg_root.set('viewBox', f'0 0 {self.width} {self.height}')
        svg_root.set('xmlns:inkscape', 'http://www.inkscape.org/namespaces/inkscape')

        title = ET.SubElement(svg_root, 'title')
        title.text = f"Body Landmark Trajectories"
        
        # Create hierarchical group structure
        layers = self.create_layers(svg_root)
        
        # Group trajectories by landmark
        landmark_groups = tracking_data.df.groupby('landmark_id')
        
        # Generate path for each landmark
        for landmark_id, group in landmark_groups:
            if landmark_id not in mp.solutions.pose.PoseLandmark:
                continue
                
            # Sort by frame to ensure proper path order
            group = group.sort_values('frame')
            
            # Convert coordinates to SVG space
            svg_coords = []
            for _, row in group.iterrows():
                svg_x, svg_y = self.normalize_to_svg_coords(row['x'], row['y'])
                svg_coords.append((svg_x, svg_y))
            
            if len(svg_coords) < 2:
                continue  # Skip if insufficient points
            
            # Get the parent path (excluding the final landmark name)
            hierarchy_parts = self.body_hierarchy[landmark_id]
            parent_path = "/".join(hierarchy_parts)
            parent_group = layers[parent_path]
            
            # Create path element in the appropriate hierarchical group
            path_elem = ET.SubElement(parent_group, 'path')
            
            # Generate path data (M for move, L for line)
            path_data = f"M {svg_coords[0][0]:.2f} {svg_coords[0][1]:.2f}"
            for x, y in svg_coords[1:]:
                path_data += f" L {x:.2f} {y:.2f}"
            
            path_elem.set('d', path_data)
            path_elem.set('stroke', self.get_landmark_color(landmark_id))
            path_elem.set('stroke-width', '1')
            path_elem.set('fill', 'none')
            path_elem.set('opacity', '0.7')
            path_elem.set('inkscape:label', mp.solutions.pose.PoseLandmark(landmark_id).name)
            path_elem.set('data-landmark', mp.solutions.pose.PoseLandmark(landmark_id).name)
        
        # Add legend if requested
        if self.show_legend:
            self.add_legend(svg_root)
        
        tree = ET.ElementTree(svg_root)
        ET.indent(tree, space="  ", level=0)
        return tree
        
    def add_legend(self, svg_root: ET.Element) -> None:
        """
        Add color legend to SVG.
        """
        legend_group = ET.SubElement(svg_root, 'g')
        legend_group.set('id', 'legend')
        
        # Legend background
        legend_bg = ET.SubElement(legend_group, 'rect')
        legend_bg.set('x', '10')
        legend_bg.set('y', '10')
        legend_bg.set('width', '200')
        legend_bg.set('height', '150')
        legend_bg.set('fill', 'white')
        legend_bg.set('stroke', 'black')
        legend_bg.set('opacity', '0.9')
        
        # Legend title
        title = ET.SubElement(legend_group, 'text')
        title.set('x', '20')
        title.set('y', '30')
        title.set('font-family', 'Arial, sans-serif')
        title.set('font-size', '14')
        title.set('font-weight', 'bold')
        title.text = 'Body Regions'
        
        # Legend entries
        legend_items = [
            ('Face', self.body_colors['face']),
            ('Left Arm', self.body_colors['left_arm']),
            ('Right Arm', self.body_colors['right_arm']),
            ('Hips', self.body_colors['hips']),
            ('Left Leg', self.body_colors['left_leg']),
            ('Right Leg', self.body_colors['right_leg'])
        ]
        
        for i, (label, color) in enumerate(legend_items):
            y_pos = 50 + i * 20
            
            # Color line
            line = ET.SubElement(legend_group, 'line')
            line.set('x1', '20')
            line.set('y1', str(y_pos))
            line.set('x2', '35')
            line.set('y2', str(y_pos))
            line.set('stroke', color)
            line.set('stroke-width', '3')
            
            # Label text
            text = ET.SubElement(legend_group, 'text')
            text.set('x', '40')
            text.set('y', str(y_pos + 4))
            text.set('font-family', 'Arial, sans-serif')
            text.set('font-size', '12')
            text.text = label
