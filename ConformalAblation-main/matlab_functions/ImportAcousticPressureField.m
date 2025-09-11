function ImportAcousticPressureField(acpr_filename)
% Import Acoustic Pressure Field 
% We first need to import the pressure acoustic fields that we simulated in
% COMSOL. 

simulation_element_size = 1e-4; % [m] Element used when interpolating acoustic field

% Import CSV exported from COMSOL
raw_pressure_field_from_comsol = csvread(acpr_filename, 9, 0);
% Strip imaginary values from solution
raw_pressure_field_from_comsol(:,3) = abs(raw_pressure_field_from_comsol(:,3));


% There are some NaN from the COMSOL export that should be replaced with 0
raw_pressure_field_from_comsol(isnan(raw_pressure_field_from_comsol)) = 0;

% Create an interpolater from the acoustic field
interpolant = scatteredInterpolant(raw_pressure_field_from_comsol(:,1), ...
                                   raw_pressure_field_from_comsol(:,2), ...
                                   raw_pressure_field_from_comsol(:,3));

% Compute properties of acoustic field (should be square)
min_x_value = min(raw_pressure_field_from_comsol(:,1));
max_x_value = max(raw_pressure_field_from_comsol(:,1));

ti = min_x_value : simulation_element_size * 7000 : max_x_value-0.1;
[qx,qy] = meshgrid(ti);
acoustic_field = interpolant(qx,qy); 

% We will pad the array up to the simulation size
% IMPORTANT: This assumes that the acoustic_field is smaller than
% simulation_width
% number_of_elements_to_pad = floor((ceil(simulation_width/1000/simulation_element_size) - length(ti))/2);
% acoustic_field = padarray(acoustic_field, [number_of_elements_to_pad number_of_elements_to_pad]);
% acoustic_field = acoustic_field*1.5;

% Plot the acoustic field
% figure
% imagesc(real(acoustic_field))
% c = colorbar;
% c.Label.String = 'Acoustic Pressure (Pa)';
% xlabel('mm')
% ylabel('mm')
% axis square

% np_filename = strrep(acpr_filename,'csv','pkl');

% mat2np(acoustic_field, np_filename, 'float64'); 

mat_filename = strrep(acpr_filename,'csv','mat');
save(mat_filename, "acoustic_field")

end