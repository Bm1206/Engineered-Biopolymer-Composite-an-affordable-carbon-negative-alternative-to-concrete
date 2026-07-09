# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 14:44:18 2026

@author: Barney
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch
from matplotlib.patches import Ellipse
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as patches
from matplotlib.image import imread
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D

# Load data
file_path = 'Map Shape Files/ne_110m_admin_0_countries.shp'
world = gpd.read_file(file_path)
data = pd.read_csv('energy_consumption_types.csv')
coordinates = pd.read_csv('coordinates.csv')
scale_factor = 0.881818 # scale for dried mass

# Merge and color assignment
world = world.merge(data, how="left", left_on="ADMIN", right_on="country")
energy_colors = {
    "Coal": 'darkgrey', "Oil": 'darkgrey', "Gas": 'violet',
    "Nuclear": 'orange', "Renewable": 'green', "Biomass": 'yellow'
}
def pvioletominant_energy(row):
    if pd.isnull(row[['Coal', 'Oil', 'Gas', 'Nuclear', 'Renewable']]).all():
        return 'grey'
    energy_data = {
        "Coal": row.get('Coal', 0), "Oil": row.get('Oil', 0),
        "Gas": row.get('Gas', 0), "Nuclear": row.get('Nuclear', 0),
        "Renewable": row.get('Renewable', 0), "Biomass": row.get('Biomass', 0)
    }
    return energy_colors.get(max(energy_data, key=energy_data.get), 'grey')
world['color'] = world.apply(pvioletominant_energy, axis=1)
world['no_data'] = world['color'] == 'grey'

# Create land mask
x = np.linspace(-180, 180, 500)
y = np.linspace(-90, 90, 250)
X, Y = np.meshgrid(x, y)
coords = np.vstack([X.ravel(), Y.ravel()]).T
land_mask = [1 if world.contains(Point(lon, lat)).any() else -999999 for lon, lat in coords]
land_mask = np.array(land_mask).reshape(X.shape)

# Load environmental data
distances_L = np.load('lignin_distance_grid.npy')
distances_C = np.load('cellulose_distance_grid.npy')

Energy_Factors = np.load('Energy_Grid_Factors.npy', allow_pickle=True)

# Calculate footprint
# Transportation + electricity + solvent - sequestration
Footprint = ((distances_L * 0.148148/1000*0.9 + distances_C * 0.148148/1000*0.1) * 0.195682 + 0.059703704 * Energy_Factors+ 0.142857 * 1.184 *(1-(0.5*0.85)) + 0.10 * 0.118 * 0.710737 *1.0 - 2.2*0.1037036 - 1.6*0.0444 )/scale_factor*1805
land_mask_with_function = land_mask * Footprint
masked_land_mask = np.ma.masked_equal(land_mask_with_function, -999999)
masked_land_mask = np.array(masked_land_mask, dtype=np.float64)

# Set up figure layout
mosaic = [
    ["a", "a", "a", "a", "a"],
    ["b",   "b",   "b",   "c",   "d"],
    ["e",   "f",   "g",   "h",   "i"],
    ["j",   "k",   "l",   "m",   "n"]
]

fig, axd = plt.subplot_mosaic(
    mosaic,
    figsize=(29, 32),
    gridspec_kw={"height_ratios": [4, 0.66, 0.66, 0.66]}
)

# Subplot (a) - map
ax_map = axd["a"]
ax_map.set_facecolor('aliceblue')
contour = ax_map.contourf(X, Y, masked_land_mask, levels=np.linspace(-400, 0, 20),
                          cmap=cm.Spectral_r, alpha=1.0)
ax_map.imshow(np.ma.masked_not_equal(land_mask, 1), extent=(-180, 180, -90, 90),
              cmap=ListedColormap(['aliceblue']), alpha=0.7)
world.boundary.plot(ax=ax_map, edgecolor='black', linewidth=1)
ax_map.set_title("a", fontweight='bold', fontsize=48, loc='left')
ax_map.set_xlabel("Longitude", fontsize=28)
ax_map.set_ylabel("Latitude", fontsize=28)
ax_map.tick_params(axis='both', which='major', labelsize=24)
ax_map.set_ylim(-60, 90)

# Cost data per country, with material being EBC and concrete representing the typical costs of concrete available on the market in the given country
cost_data = {
    'China': {'material': 212, 'concrete': 240},
    'United States of America': {'material': 289, 'concrete': 408},
    'Canada': {'material': 302, 'concrete': 400},
    'Japan': {'material': 270, 'concrete': 576},
    'France': {'material': 339, 'concrete': 552},
    'Brazil': {'material': 335, 'concrete': 240},
    'India': {'material': 232, 'concrete': 233},
    'Australia': {'material': 426, 'concrete': 500},
    'South Africa': {'material': 269, 'concrete': 170},

}

countries_to_shift = ['United States of America', 'Canada', 'China', 'India', 'South Africa']
adjust_lon = [20, -10, 0, -2, -1]  # example shifts in degrees longitude
adjust_lat = [-10, -8, -7, 0, -3]    # example shifts in degrees latitude

max_cost = max(max(v.values()) for v in cost_data.values())
bar_max_height = 25  # max bar height in degrees latitude
bar_width = 5        # width of each bar

for country_name, costs in cost_data.items():
    country_geom = world[world['ADMIN'] == country_name]
    if country_geom.empty:
        continue
    centroid = country_geom.geometry.centroid.values[0]
    lon, lat = centroid.x, centroid.y
    
    # Apply shift only for bars of specified countries
    if country_name in countries_to_shift:
        idx = countries_to_shift.index(country_name)
        lon += adjust_lon[idx]
        lat += adjust_lat[idx]
    
    # Heights normalized
    height_material = (costs['material'] / max_cost) * bar_max_height
    height_concrete = (costs['concrete'] / max_cost) * bar_max_height
    
    # Positions for bars side-by-side
    left_bar_x = lon - bar_width / 2  
    right_bar_x = lon + bar_width / 2 
    
    # darkgrey bar (material cost)
    rect_material = plt.Rectangle(
        (left_bar_x, lat),
        bar_width,
        height_material,
        linewidth=1,
        edgecolor='black',
        facecolor='violet',
        alpha=0.7,
        zorder=20
    )
    ax_map.add_patch(rect_material)
    
    # violet bar (concrete cost)
    rect_concrete = plt.Rectangle(
        (right_bar_x, lat),
        bar_width,
        height_concrete,
        linewidth=1,
        edgecolor='black',
        facecolor='darkgrey',
        alpha=0.7,
        zorder=20
    )
    ax_map.add_patch(rect_concrete)
    
    # Material cost label with transparent background
    ax_map.text(
        left_bar_x - bar_width / 2, lat + height_material + 0.5,
        f"${costs['material']}", ha='center', va='bottom',
        fontsize=16, fontweight='bold', color='black', zorder=6,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
    )
    
    # Concrete cost label with transparent background
    ax_map.text(
        right_bar_x + bar_width, lat + height_concrete + 0.5,
        f"${costs['concrete']}", ha='center', va='bottom',
        fontsize=16, fontweight='bold', color='black', zorder=6,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, boxstyle='round,pad=0.2')
    )


# Define legend elements for material and concrete bars
legend_elements = [
    Patch(facecolor='violet', edgecolor='black', label=r'EBC [$\rho$ = 1805 kg/m³, $\sigma$ = 35 MPa]'),
    Patch(facecolor='darkgrey', edgecolor='black', label=r'Concrete [$\rho$ = 2400 kg/m³, $\sigma$ = 20 MPa]')
]

# Add the legend to the bottom right of the figure
fig.legend(handles=legend_elements,
           loc='lower right',
           bbox_to_anchor=(0.965, 0.45),  # (x, y) in figure coordinates
           fontsize=30, ncol=1, title='Cost of materials [$/m³] ', title_fontsize=36,
           frameon=True)

# Colorbar
cbar_ax = fig.add_axes([0.05, 0.5, 0.5, 0.02])
ticks = np.arange(-400, 0, 50)
ticks = np.append(ticks, 0)

cbar = fig.colorbar(contour, cax=cbar_ax, orientation='horizontal', ticks=ticks)

cbar.set_label('Carbon footprint [kg $CO_2$/ $m^3$]', fontsize=28)
cbar.ax.tick_params(labelsize=22)

inset_ax = inset_axes(ax_map,
                      width="50%",  
                      height="50%",  
                      bbox_to_anchor=(-155, -40, 100, 100),  # lon_min, lat_min, width_lon, height_lat
                      bbox_transform=ax_map.transData,
                      loc='lower left')

# Example data for Ashby plot (density, strength, width, height)
materials_ashby = {
    'EBC (This Study)': {
        'density': 1800, 'strength': 29,
        'density_spread': 300, 'strength_spread': 8,
        'color': 'violet'
    },
    'Concrete': {
        'density': 2400, 'strength': 18,
        'density_spread': 400, 'strength_spread': 8,
        'color': 'darkgrey'
    },
    'Clay Brick': {
        'density': 1900, 'strength': 15,
        'density_spread': 200, 'strength_spread': 5,
        'color': 'orangered'
    },
    'Asphalt': {
        'density': 2300, 'strength': 5,
        'density_spread': 300, 'strength_spread': 2,
        'color': 'black'
    },
    'Hempcrete': {
        'density': 500, 'strength': 4.4,
        'density_spread': 100, 'strength_spread': 2.5,
        'color': 'darkkhaki'
    },
    'Lignin BSC': {
        'density': 1350, 'strength': 6,
        'density_spread': 150, 'strength_spread': 4,
        'color': 'steelblue'
    }
}

inset_ax.axhline(
    y=17,
    color='black',
    linestyle='--',
    linewidth=2,
    label='17 MPa threshold'
)

# Plot ellipses
for name, props in materials_ashby.items():
    ellipse = Ellipse(
        (props['density'], props['strength']),
        width=props['density_spread'],
        height=props['strength_spread'],
        facecolor=props['color'],
        edgecolor='black',
        alpha=0.7,
        label=name
    )
    inset_ax.add_patch(ellipse)
    # Add material label above the ellipse
    inset_ax.text(
        props['density'],
        props['strength'] + props['strength_spread'] * 0.6,
        name,
        ha='center',
        fontsize=14,
        color='black'
    )
    
# Axes settings
inset_ax.set_xlim(0, 3000)
inset_ax.set_ylim(0, 50)
inset_ax.set_xlabel("Density [kg/m³]", fontsize=20)
inset_ax.set_ylabel("Compressive Strength [MPa]", fontsize=20)
inset_ax.tick_params(axis='both', which='major', labelsize=20)
inset_ax.grid(True, linestyle=':', alpha=0.5)

materials = ['EBC', 'Concrete']

bar_data = [
    [21.09963186, 48.7224, 0.602847, 2.43612],
    [15.06794679, 75.54, 0.430513, 3.777],
    [0.511877315, 1.0488, 0.014625, 0.05244],
    [2.45583E-05, 0.00022176, 7.02E-07, 0.000011088],
    [0.336140391, 0.8784, 0.009604, 0.04392],
    [90367.28235, 562466.16, 2581.922, 28123.308],
    [7140.539615, 195279.48, 204.0154, 9763.974],
    [10.78024962, 32.1816, 0.308007, 1.60908],
    [39.95551552, 591.2808, 1.141586, 29.56404],  
    [2.956310874, 5.58, 0.084466, 0.279],  
    [0.082449159, 0.2856, 0.002356, 0.01428],  
    [-322.0880802, 501.6977, -9.20252, 25.08], 
]

bar_titles = ['c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']

# Update color mapping
material_colors = {
    'EBC': 'violet',
    'Concrete': 'darkgrey'
}

bar_width = 0.6
x_left = np.array([0, 1])
x_right = np.array([3, 4])

subplot_xlabels = [
    'Carcinogens', 'Non-carcinogens', 
    'Respiratory inorganics', 'Ozone layer depletion', 
    'Respiratory organics', 'Aquatic ecotoxicity',
    'Terrestrial ecotoxicity', 'Terrestrial acid',
    'Land occupation', 'Aquatic acidification',
    'Aquatic eutrophication', 'Global warming' 
]

subplot_unitlabels = [
    'kg C2H3Cl', 'kg C2H3Cl', 
    'kg C2H4', '$10^{-7}$ kg CFC-11', 
    'kg C2H4', 'kg TEG water',
    'kg TEG soil', 'kg SO2',
    '$m^2$ arable land', 'kg SO2 eq',
    'kg PO4 P-lim', 'kg CO2' 
]


def annotate_reduction(ax, x_ebc, y_ebc, x_conc, y_conc):
    if y_conc == 0:
        return

    reduction = (1 - y_ebc / y_conc) * 100

    # Dashed horizontal line at concrete height
    ax.hlines(
        y_conc,
        x_ebc - 0.3,
        x_conc + 0.3,
        colors='black',
        linestyles='dashed',
        linewidth=1.5,
        zorder=2
    )

    ax.annotate(
        '',
        xy=(x_ebc, y_ebc),     # bottom (EBC)
        xytext=(x_ebc, y_conc), # top (Concrete)
        arrowprops=dict(
            arrowstyle='->',
            linestyle='--',
            color='black',
            lw=1.5
        ),
        zorder=4
    )

    # Label slightly above midpoint
    y_mid = (y_ebc + y_conc) / 2

    ax.text(
        x_ebc+0.1,
        y_mid * 1.08,
        f"-{reduction:.0f}%",
        ha='center',
        va='bottom',
        fontsize=16,
        color='green',
        fontweight='normal',
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor='lightgrey',
            edgecolor='none',
            alpha=0.4  
        ),
        zorder=10
    )
    
def safe_ylim(vals, pad=1.4):
    vmin = np.min(vals)
    vmax = np.max(vals)

    if vmin >= 0:
        return 0, vmax * pad
    else:
        return vmin * pad, vmax * pad
    
for i in range(12):
    row = 1 + i // 5  
    col = i % 5      
    ax_left = axd[bar_titles[i]]
    ax_right = ax_left.twinx()

    # Split left and right values
    left_vals = bar_data[i][:2]
    right_vals = bar_data[i][2:]

    # Plot left group bars
    for idx, val in enumerate(left_vals):
        material = materials[idx]
        ax_left.bar(x_left[idx], val, color=material_colors[material], width=bar_width, zorder=3)
        
        annotate_reduction(
            ax_left,
            x_left[0], left_vals[0],   # EBC
            x_left[1], left_vals[1]    # Concrete
        )
        
    ax_left.set_ylim(*safe_ylim(left_vals))

    ax_left.tick_params(axis='y', labelsize=24)

    # Plot right group bars
    for idx, val in enumerate(right_vals):
        material = materials[idx]
        ax_right.bar(x_right[idx], val, color=material_colors[material], width=bar_width, zorder=3)
        
        annotate_reduction(
            ax_right,
            x_right[0], right_vals[0],  # EBC
            x_right[1], right_vals[1]   # Concrete
        )
        
    if bar_titles[i] == "n":

        ax_left.hlines(
            y=0,
            xmin=x_left[0] - 1.0,
            xmax= x_left[1] + 1.0,
            color='black',
            linewidth=2.5,
            linestyle='-',
            zorder=2
        )
    
        ax_right.hlines(
            y=0,
            xmin=2,                 
            xmax=x_right[1] + 1.0, 
            color='black',
            linewidth=2.5,
            linestyle='-',
            zorder=2
        )
        
    ax_right.set_ylim(*safe_ylim(right_vals))
    ax_right.tick_params(axis='y', labelsize=24)

    # Dashed vertical line between groups
    ax_left.axvline(x=2, linestyle='--', color='gray', linewidth=1.5, zorder=1)

    # Set ticks and labels
    xticks_combined = list(x_left) + list(x_right)
    xticklabels_combined = materials + materials
    ax_left.set_xticks(xticks_combined)
    ax_left.set_xticklabels(xticklabels_combined, rotation=45, fontsize=20)

    ax_right.set_xticks([])
    ax_right.set_xticklabels([])

    # Subplot title
    ax_left.set_title(bar_titles[i], fontweight='bold', fontsize=40, loc='left')
    ax_left.set_xlabel(str(subplot_xlabels[i]), fontsize=24, labelpad=10)
    ax_left.grid(True, axis='y', linestyle=':', alpha=0.5)

    ax_left.set_ylabel(str(subplot_unitlabels[i]) + "/ $m^3$", fontsize=24)
    ax_right.set_ylabel(str(subplot_unitlabels[i]) + "/ $m^3$MPa", fontsize=24)
        
    # Create formatter
    formatter_left = ScalarFormatter(useMathText=True)
    formatter_left.set_scientific(True)
    formatter_left.set_powerlimits((-2, 3))  
    
    formatter_right = ScalarFormatter(useMathText=True)
    formatter_right.set_scientific(True)
    formatter_right.set_powerlimits((-2, 3))
    
    # Apply to both y-axes
    ax_left.yaxis.set_major_formatter(formatter_left)
    ax_right.yaxis.set_major_formatter(formatter_right)
    
    # Change offset (scientific notation multiplier) font size
    ax_left.yaxis.offsetText.set_fontsize(22)
    ax_right.yaxis.offsetText.set_fontsize(22)
        
    if i < 5:
        category = "HH"
        cat_color = "saddlebrown"   
    elif i < 11:
        category = "EH"
        cat_color = "steelblue"     
    else:
        category = "GW"
        cat_color = "darkgreen"     
    
    ax_left.text(
        0.97, 0.95,
        category,
        transform=ax_left.transAxes,
        ha='right',
        va='top',
        fontsize=20,
        fontweight='bold',
        color=cat_color,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor='#f2f2f2',  # pale background
            edgecolor=cat_color,  # matches text
            linewidth=1.5,
            alpha=0.9
        ),
        zorder=10
    )

legend_patches = [
    Patch(facecolor=material_colors['EBC'], edgecolor='black', label='Engineered biopolymer composite [EBC]'),
    Patch(facecolor=material_colors['Concrete'], edgecolor='black', label='Concrete')
]


fig.legend(
    handles=legend_patches,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    fontsize=24,
    ncol=2,
    frameon=True,
    title='Environmental and Human health impacts from material production ', title_fontsize=26
)


world['has_data'] = ~world[['Coal', 'Oil', 'Gas', 'Nuclear', 'Renewable', 'Biomass']].isna().all(axis=1)

exclude_from_hatching = ['Russian Federation']

world[
    (~world['has_data']) &
    (~world['ADMIN'].isin(exclude_from_hatching))
].plot(
    ax=ax_map,
    facecolor='whitesmoke',
    hatch='///',
    edgecolor='black',
    linewidth=0.5,
    zorder=10
)

import matplotlib.patches as mpatches

no_data_patch = mpatches.Patch(
    facecolor='lightgrey',
    hatch='///',
    edgecolor='black',
    label='No data'
)
ax_map.legend(handles=[no_data_patch], loc='upper left', fontsize=26)

ax_b = axd["b"]

# remove axes
ax_b.axis("off")

# load image
img = imread("Person_Health.png")

# create icon
imagebox = OffsetImage(img, zoom=0.65)   # adjust zoom as needed

# place icon in subplot
ab = AnnotationBbox(
    imagebox,
    (0.05, 0.70),              # center of subplot
    xycoords='axes fraction',
    frameon=False
)

ax_b.add_artist(ab)

# Green reduction text box
ax_b.text(
    0.235, 0.35,
    "-57% Human\n Health (HH)\nimpacts",
    transform=ax_b.transAxes,
    ha='center',
    va='center',
    fontsize=24,
    fontweight='normal',
    color='brown'
)

img = imread("Tree_Icon.png")

# create icon
imagebox = OffsetImage(img, zoom=0.75)   # adjust zoom as needed

# place icon in subplot
ab = AnnotationBbox(
    imagebox,
    (0.50, 0.70),              # center of subplot
    xycoords='axes fraction',
    frameon=False
)

# Green reduction text box
ax_b.text(
    0.74, 0.35,
    "-95% Ecosystem\n Health (EH)\nimpacts",
    transform=ax_b.transAxes,
    ha='center',
    va='center',
    fontsize=24,
    fontweight='normal',
    color='darkblue'
)

ax_b.add_artist(ab)

img = imread("Temp_Icon.png")

# create icon
imagebox = OffsetImage(img, zoom=0.40)   # adjust zoom as needed

# place icon in subplot
ab = AnnotationBbox(
    imagebox,
    (0.865, 0.65),              # center of subplot
    xycoords='axes fraction',
    frameon=False
)

ax_b.add_artist(ab)

# Green reduction text box
ax_b.text(
    0.975, 0.35,
    "-144% Global\n Warming (GW)\nimpact",
    transform=ax_b.transAxes,
    ha='center',
    va='center',
    fontsize=24,
    fontweight='normal',
    color='Green'
)

# Green reduction text box
ax_b.text(
    0.0, 1.45,
    "b",
    transform=ax_b.transAxes,
    ha='center',
    va='center',
    fontsize=40,
    fontweight='bold',
    color='black'
)


import matplotlib.patches as patches

rect = patches.Rectangle(
    (0.01, 0.30),   
    0.60 - 0.01,              
    0.16,                    
    transform=fig.transFigure,
    facecolor='lightblue',
    alpha=0.3,
    edgecolor='none',
    zorder=0
)

fig.add_artist(rect)

rect = patches.Rectangle(
    (0.01, 0.285),   
    0.60 - 0.01,             
    0.0155,                    
    transform=fig.transFigure,
    facecolor='lawngreen',
    alpha=0.3,
    edgecolor='none',
    zorder=0
)

fig.add_artist(rect)

line = Line2D(
    [0.01, 0.60],   
    [0.30, 0.30],   
    transform=fig.transFigure,
    color='black',
    linewidth=3
)

fig.lines.append(line)

# Final layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.25)
plt.savefig('Figure5.jpg', dpi=300, bbox_inches='tight', pad_inches=0.03, pil_kwargs={"compress_level": 9})
plt.show()

