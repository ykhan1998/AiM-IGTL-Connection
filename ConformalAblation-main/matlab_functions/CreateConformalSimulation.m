function [kdiff, kgrid, Q] = CreateConformalSimulation(acpr_filename, init_temp, cem_temp)
%% Import Acoustic Pressure Field 
% We first need to import the pressure acoustic fields that we simulated in COMSOL. 

load(acpr_filename); % acoustic_field

%% Simulation Settings
% IMPORTANT! Do not forget to specify settings here!

%%%%%%%% ULTRASOUND RELATED & MATERIAL PROPERTIES
ultrasound_excitation_frequency = 5.2373e6; % [Hz]

alpha_coeff_in_phantom = 0.53; % [dB/(MHz^y cm)]
% alpha_coeff_in_phantom = 0.5035; % [dB/(MHz^y cm)]
% alpha_coeff_in_phantom = 0.5565; % [dB/(MHz^y cm)]
% alpha_coeff_in_phantom = 0.4505; % [dB/(MHz^y cm)]
% alpha_coeff_in_phantom = 0.6095; % [dB/(MHz^y cm)]

sound_speed = 1551; % [m/s]
% sound_speed = 1473; % [m/s]
% sound_speed = 1628; % [m/s]

density_in_phantom = 1058; % [kg/m^3]

thermal_conductivity_of_phantom = 0.5367;% [W/(m.K)]
% thermal_conductivity_of_phantom = 0.5;% [W/(m.K)]
% thermal_conductivity_of_phantom = 0.56;% [W/(m.K)]

specific_heat_of_phantom = 3451; % [J/(kg.K)]
% specific_heat_of_phantom = 3278; % [J/(kg.K)]
% specific_heat_of_phantom = 3623; % [J/(kg.K)]

initial_temperature = init_temp; % [C]
number_of_elements = length(acoustic_field);

%%%%%%%% SIZE RELATED
probe_radius = 0.75e-3; % [m] Probe is always assumed to be in the center of simulation
simulation_element_size = 7e-4; % [m] Element used when interpolating acoustic field

% %%%%%%% Truncate Pressure Field to Match Sim Dimensions
% % NOTE: Make sure this math does not yield decimals
% size_delta_from_default = 143 - envConstants.simulation_width_pixels;
% half_size_delta = size_delta_from_default / 2;
% acoustic_field = acoustic_field(half_size_delta:half_size_delta+envConstants.simulation_width_pixels-1, ...
%                                 half_size_delta:half_size_delta+envConstants.simulation_width_pixels-1);

%% Setup the KWaveGrid Simulation Environment

% Our thermal simulation should be sqrt(2) smaller than the acoustic field
% size since the acoustic field size will be rotated and its diagonal
% should fit within the simulation.

% Create the computational grid
kgrid = kWaveGrid(number_of_elements, simulation_element_size, ...
                  number_of_elements, simulation_element_size);

% Create a circle representing the probe
center = number_of_elements/2;
[columnsInImage, rowsInImage] = meshgrid(1:number_of_elements);
circle_mask = (rowsInImage - center).^2 + (columnsInImage - center).^2 <= (probe_radius/simulation_element_size).^2; %Probe Mask

% Setup the medium
medium.sound_speed = sound_speed; % Same everywhere
medium.density = density_in_phantom; % Same everywhere
medium.alpha_coeff = alpha_coeff_in_phantom; % Same everywhere
medium.alpha_power = 1; % Same everywhere
medium.thermal_conductivity = thermal_conductivity_of_phantom; %* ones(number_of_elements); % Same everywhere
medium.specific_heat = specific_heat_of_phantom * ones(number_of_elements); % Same everywhere
medium.specific_heat(circle_mask) = 1e12; % In probe

%Calculate the time step using an integer number of points per period
ppw = sound_speed / (ultrasound_excitation_frequency * simulation_element_size); % points per wavelength
cfl = 0.3;                                                              % cfl number
ppp = ceil(ppw / cfl);                                                  % points per period
T   = 1 / ultrasound_excitation_frequency;                              % period [s]
dt  = T / ppp;                                                          % time step [s]

% Calculate the number of time steps to reach steady state
t_end = sqrt( kgrid.x_size.^2 + kgrid.y_size.^2 ) / sound_speed; 
Nt = round(t_end / dt);

% Create the time array
kgrid.setTime(Nt, dt);              
              
% Convert the absorption coefficient to nepers/m
alpha_np = db2neper(medium.alpha_coeff, medium.alpha_power) * (2 * pi * ultrasound_excitation_frequency).^medium.alpha_power;
                          
% Compute volume rate of heat deposition (basically the amount of heating due to ultrasound)
Q = alpha_np .* acoustic_field.^2 ./ (medium.density .* medium.sound_speed);

% Create source
source.Q = Q;
source.T0 = initial_temperature;

%% Create the kWaveDiffusion object
kdiff = kWaveDiffusion(kgrid, medium, source, cem_temp);

end

