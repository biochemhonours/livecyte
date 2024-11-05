#!/usr/bin/env python
# coding: utf-8

# In[86]:


### IMPORT NECESSARY LIBRARIES ############################################################################################

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import (
    FloatSlider, Dropdown, Button, SelectMultiple, Text, VBox, HBox, Output, Layout, GridspecLayout, Checkbox, Label
)
from IPython.display import display, HTML
import collections

### INITIALIZE GLOBAL VARIABLES
wells_data = collections.defaultdict(list)
combined_df = pd.DataFrame()
population_selector = None
metadata_dict = {}
output_area = Output()
metadata_area = Output()
csv_filename = 'combined_dataframe.csv'  # Filename for the CSV with metadata assigned

### DIRECTORY ############################################################################################

def load_data(directory):
    global combined_df, population_selector, metadata_dict

    # Clear previous data
    wells_data.clear()
    combined_df = pd.DataFrame()
    metadata_dict = {}

    # Loop through the files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            
            try:
                # Read the data
                df = pd.read_csv(file_path, header=None, on_bad_lines='skip')  # Skip bad lines
                
                # Extract well identifier from cell D1 (which is at index [0, 3])
                well_id = df.iloc[0, 3]
                
                # If well_id contains an underscore, take the part before it
                well_id = well_id.split('_')[0]  # Get just the well identifier
                
                # Read the actual data, skipping the first row which is now used for well ID
                df = pd.read_csv(file_path, skiprows=1, on_bad_lines='skip')  # Skip bad lines
                df['Well'] = well_id  # Add a new column with the well ID
                
                # Ensure the well_id is a key in wells_data
                if well_id not in wells_data:
                    wells_data[well_id] = []  # Initialize list if not already present
                
                wells_data[well_id].append(df)  # Append the DataFrame to the corresponding well
            except Exception as e:
                with output_area:
                    output_area.clear_output()
                    print(f"Error processing file {filename}: {e}")


    # Now concatenate the DataFrames for each well
    for well_id, dfs in wells_data.items():
        combined_well_data = pd.concat(dfs, ignore_index=True)
        combined_df = pd.concat([combined_df, combined_well_data], ignore_index=True)

    # Create population selector
    population_selector = SelectMultiple(
        options=list(combined_df['Well'].unique()),
        value=list(combined_df['Well'].unique()),
        description='Wells:', 
        disabled=False,
    )

    # Display the widgets for plotting after loading data
    display_widgets()


# Directory input and load button
directory_input = Text(
    description='Directory:',
    value='/Users/marcinmaniak/Desktop/PhD/Livecyte feature tables/261024 - NewSegmentation_041124',  # Default directory
    layout=Layout(width='800px')  # Set the width as desired
)

load_data_button = Button(description="Load Data")
load_data_button.on_click(lambda b: load_data(directory_input.value))

# Display widgets for directory input and load data
display(VBox([directory_input, load_data_button, output_area]))

    
### UX/WIDGETS ############################################################################################

def display_widgets():
    global y_min_slider, y_max_slider, point_size_slider, variable_dropdown, stat_dropdown
    global violin_checkbox, beeswarm_checkbox, include_assigned_only_checkbox, metadata_area

    # Metadata input area
    metadata_area = Output()
    create_metadata_schematic()

    # Plotting sliders and dropdowns
    y_min_slider = FloatSlider(value=0, min=0, max=500, step=10, description='Y-axis Min')
    y_max_slider = FloatSlider(value=4500, min=0, max=10000, step=10, description='Y-axis Max')
    point_size_slider = FloatSlider(value=2, min=0.5, max=10, step=0.5, description='Point Size')

    variable_dropdown = Dropdown(
        options=['Mean Area (µm²)', 'Mean Volume (µm³)', 'Mean Mean Thickness (µm)', 'Mean Dry Mass (pg)', 'Mean Sphericity ()'],
        value='Mean Area (µm²)',
        description='Variable:'
    )

    stat_dropdown = Dropdown(
        options=['Mean ± SD', 'Median ± IQR'],
        value='Mean ± SD',
        description='Statistics:'
    )

    violin_checkbox = Checkbox(value=True, description='Show Violin Plot')
    beeswarm_checkbox = Checkbox(value=True, description='Show Beeswarm Plot')

    include_assigned_only_checkbox = Checkbox(value=False, description='Include Only Assigned Wells')

    # Create the plot button
    plot_button = Button(description="Plot Data")
    plot_button.on_click(lambda b: interactive_overlay_plot(
        selected_populations=list(population_selector.value),
        selected_variable=variable_dropdown.value,
        y_min=y_min_slider.value,
        y_max=y_max_slider.value,
        show_violin=violin_checkbox.value,
        show_beeswarm=beeswarm_checkbox.value,
        stat_option=stat_dropdown.value,
        point_size=point_size_slider.value
    ))

    # Create the show DataFrame button
    show_dataframe_button = Button(description="Show DataFrame")
    show_dataframe_button.on_click(display_dataframe)

    # Create the download button
    download_button = Button(description="Download CSV")
    download_button.on_click(lambda b: download_csv())
    

    # New area for extra populations metadata input
    extra_population_area = Output()
    create_extra_population_input(extra_population_area)

    # Create section headers
    header_wells = Label(value="1) Choose wells for analysis:", layout=Layout(margin='10px 0 0 0'))
    header_metadata_24 = Label(value="2a) Assign Metadata (if 24-well plate):", layout=Layout(margin='10px 0 0 0'))
    header_metadata_else = Label(value="2b) Assign Metadata (else):", layout=Layout(margin='10px 0 0 0'))
    header_threshold = Label(value="3) Gating - Set Thresholds:", layout=Layout(margin='10px 0 0 0'))
    header_plotting = Label(value="4) Plotting Options:", layout=Layout(margin='10px 0 0 0'))
    header_download = Label(value="5) Data Management:", layout=Layout(margin='10px 0 0 0'))
    
    
    # Create sections
    
    population_selector_section = VBox([
        header_wells,
        population_selector,

    ])
    
    metadata_section_24 = VBox([
        header_metadata_24,
        metadata_area,

    ])
    
    metadata_section_else = VBox([
        header_metadata_else,

        extra_population_area,  # New area for extra populations
        include_assigned_only_checkbox
    ])
        
    plotting_options = VBox([
        header_plotting,
        HBox([variable_dropdown, stat_dropdown]),
        HBox([y_min_slider, y_max_slider, point_size_slider]),
        HBox([violin_checkbox, beeswarm_checkbox]),
        plot_button
    ])

    thresholding_options = VBox([
        header_threshold,
        HBox([area_threshold_slider, volume_threshold_slider, thickness_threshold_slider]),
    ])
    
    data_management_section = VBox([
        header_download,
        HBox([show_dataframe_button, download_button])
    ])

    # Combine all sections into a main layout
    main_layout = VBox([
        population_selector_section,
        metadata_section_24,
        metadata_section_else,
        thresholding_options,
        plotting_options,
        data_management_section,
        output_area
    ])

    # Display the main layout
    display(main_layout)
    
### CREATE METADATA ############################################################################################

# Function to create a metadata input schematic for the 24-well plate
def create_metadata_schematic():
    grid = GridspecLayout(4, 6, layout=Layout(grid_gap='10px'))
    well_labels = [f"{row}{col}" for row in ['A', 'B', 'C', 'D'] for col in range(1, 7)]
    
    for i, well in enumerate(well_labels):
        text_widget = Text(
            value=metadata_dict.get(well, ''),
            placeholder='Enter metadata',
            description=well,
            disabled=False,
        )
        text_widget.observe(lambda change, well=well: update_metadata(change, well), names='value')
        grid[i // 6, i % 6] = text_widget  # Place the text widget in the grid
    
    with metadata_area:
        metadata_area.clear_output()
        display(grid)
        


def update_metadata(change, well):
    metadata_dict[well] = change['new']
    
def create_extra_population_input(output_area):
    # Create a Text widget for entering additional populations
    population_input = Text(
        description='Populations:',
        placeholder='Enter population ID (comma separated)',
        layout=Layout(width='400px')  # Set the width as desired
    )
    
    # Create a Text widget for entering corresponding metadata
    metadata_input = Text(
        description='Metadata:',
        placeholder='Enter metadata for populations (comma separated)',
        layout=Layout(width='400px')  # Set the width as desired
    )
    
    # Button to assign metadata to the additional populations
    assign_button = Button(description="Assign Metadata")
    assign_button.on_click(lambda b: assign_extra_population_metadata(
        population_input.value,
        metadata_input.value,
        output_area
    ))

    # Display the input fields and button
    with output_area:
        output_area.clear_output()
        display(population_input, metadata_input, assign_button)

def assign_extra_population_metadata(population_ids, metadata_values, output_area):
    ids = [id.strip() for id in population_ids.split(',') if id.strip()]
    values = [value.strip() for value in metadata_values.split(',') if value.strip()]

    if ids and values:
        if len(ids) == len(values):  # Ensure both lists are the same length
            for population_id, metadata_value in zip(ids, values):
                metadata_dict[population_id] = metadata_value
            with output_area:
                output_area.clear_output()
                print(f'Metadata assigned for populations: {", ".join(ids)}')
        else:
            with output_area:
                output_area.clear_output()
                print("Please ensure the number of population IDs matches the number of metadata entries.")
    else:
        with output_area:
            output_area.clear_output()
            print("Please enter valid population IDs and metadata.")

### DATA ANALYSIS - GATING & METADATA ASSIGNMENT  #########################################################################################

# Gating thresholds
area_threshold_slider = FloatSlider(value=150, min=0, max=1000, step=10, description='Area')
volume_threshold_slider = FloatSlider(value=0, min=0, max=1000, step=10, description='Volume')
thickness_threshold_slider = FloatSlider(value=0, min=0, max=10, step=0.1, description='Thickness')


# Update generate_filtered_dataframe to accept slider values as parameters
def generate_filtered_dataframe(area_threshold, volume_threshold, thickness_threshold):
    filtered_df = combined_df[
        (combined_df['Well'].isin(list(population_selector.value))) &
        (combined_df['Mean Area (µm²)'] > area_threshold) &
        (combined_df['Mean Volume (µm³)'] > volume_threshold) &
        (combined_df['Mean Mean Thickness (µm)'] > thickness_threshold)
    ]

    # Create a mapping of wells to metadata, including extra populations
    metadata_map = {well: metadata_dict.get(well, '') for well in filtered_df['Well'].unique()}
    filtered_df['Metadata'] = filtered_df['Well'].map(metadata_map)

    if include_assigned_only_checkbox.value:
        filtered_df = filtered_df[filtered_df['Metadata'] != '']

    return filtered_df


### DISPLAY DATAFRAME #########################################################################################

# Update display_dataframe to pass slider values when calling generate_filtered_dataframe
def display_dataframe(b):
    with output_area:
        output_area.clear_output()
        df_with_metadata = generate_filtered_dataframe(
            area_threshold_slider.value, 
            volume_threshold_slider.value, 
            thickness_threshold_slider.value
        )
        display(df_with_metadata)
        
### DOWNLOAD DATAFRAME #########################################################################################

# Function to download the filtered DataFrame with metadata as a CSV file
def download_csv():
    df_with_metadata = generate_filtered_dataframe(
        area_threshold_slider.value, 
        volume_threshold_slider.value, 
        thickness_threshold_slider.value
        )
    with output_area:
        output_area.clear_output()
        if not df_with_metadata.empty:
            df_with_metadata.to_csv(csv_filename, index=False)
            print(f'Data saved to {csv_filename}.')
        else:
            print('No data to download. Please adjust the filters.')
            
### PLOTTTING #########################################################################################
            
# Update interactive_overlay_plot to pass slider values when calling generate_filtered_dataframe
def interactive_overlay_plot(selected_populations, 
                             selected_variable='Mean Area (µm²)', 
                             y_min=0, y_max=4500, 
                             show_violin=True, show_beeswarm=True, 
                             stat_option='Mean ± SD', point_size=2):
    # Use current threshold slider values
    filtered_df = generate_filtered_dataframe(
        area_threshold_slider.value, 
        volume_threshold_slider.value, 
        thickness_threshold_slider.value
    )

    with output_area:
        output_area.clear_output()
        if filtered_df.empty:
            print("No data available for the selected criteria.")
            return
    
    plt.figure(figsize=(12, 8))
    
    # Violin plot overlay
    if show_violin:
        sns.violinplot(
            x='Metadata',
            y=selected_variable,
            data=filtered_df, 
            palette="pastel",  
            scale='width',  
            width=0.6,
            inner=None,
        )
    
    # Beeswarm plot overlay
    if show_beeswarm:
        sns.swarmplot(
            x='Metadata',
            y=selected_variable,
            data=filtered_df,
            color='k',
            size=point_size
        )

    # Calculate and plot statistics at exact x-axis positions
    cell_counts = filtered_df['Metadata'].value_counts()
    metadata_categories = filtered_df['Metadata'].unique()
    for i, identifier in enumerate(metadata_categories):
        category_data = filtered_df[filtered_df['Metadata'] == identifier][selected_variable]
        count = cell_counts.get(identifier, 0)
        
        if not category_data.empty:
            x_position = i  # Use exact x-axis position from the plot
            if stat_option == 'Mean ± SD':
                mean = category_data.mean()
                std_dev = category_data.std()
                plt.errorbar(x_position, mean, yerr=std_dev, fmt='o', color='black', capsize=10, elinewidth=2, capthick=2)
                plt.text(x_position, y_max * 0.9, f'n={count}\n{mean:.2f} ± {std_dev:.2f}', 
                         ha='center', va='top', fontsize=10, color="black")
            elif stat_option == 'Median ± IQR':
                median = category_data.median()
                iqr = category_data.quantile(0.75) - category_data.quantile(0.25)
                plt.errorbar(x_position, median, yerr=iqr / 2, fmt='o', color='black', capsize=10, elinewidth=2, capthick=2)
                plt.text(x_position, y_max * 0.9, f'n={count}\n{median:.2f} ± {iqr/2:.2f}', 
                         ha='center', va='top', fontsize=10, color="black")
    plt.ylim(y_min, y_max)
    plt.title(f'{selected_variable} by Metadata')
    plt.ylabel(selected_variable)
    plt.xlabel('Metadata')
    plt.xticks(range(len(metadata_categories)), metadata_categories)
    plt.grid(True)
    plt.show()


# In[ ]:




