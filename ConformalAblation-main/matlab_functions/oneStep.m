function [tempMap] = oneStep(rotate_angle, power_switch, dt)
    global kdiff  kgrid  Q

    if power_switch == false
        kdiff.Q = zeros(size(kdiff.Q));
    else
        kdiff.Q = imrotate(kdiff.Q, double(rotate_angle), 'bilinear', 'crop');
    end

    % take Nt time steps of size dt, check kWaveDiffusion.m
    nt = 1;
    kdiff.takeTimeStep(nt, dt);
    tempMap = kdiff.T;
    
    % lesionMap = kdiff.lesion_map;
    % cem_map = kdiff.cem43;
    % lesionMap = cem >= cem_threshold;
end