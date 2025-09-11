function resetTissue(acpr_filename, init_temp, cem_temp)
    global kdiff  kgrid  Q
    [kdiff, kgrid, Q] = CreateConformalSimulation(acpr_filename, init_temp, cem_temp);
    % disp("Tissue is reset.")
end