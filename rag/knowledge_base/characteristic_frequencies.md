# Characteristic bearing frequencies

With shaft frequency fr = rpm / 60, n rolling elements, ball diameter d, pitch diameter D and contact angle theta:

FTF (cage) = fr / 2 * (1 - d/D * cos(theta)).
BPFO (ball pass frequency, outer race) = n * fr / 2 * (1 - d/D * cos(theta)).
BPFI (ball pass frequency, inner race) = n * fr / 2 * (1 + d/D * cos(theta)).
BSF (ball spin frequency) = D / (2 d) * fr * (1 - (d/D * cos(theta))^2).

Real bearings slip, so the measured peak can differ from the calculated value by a small percentage; the project
therefore searches a tolerance band around each frequency and its harmonics. A peak at BPFO points to the outer race,
at BPFI to the inner race, at BSF (or twice BSF) to a rolling element, at FTF to a cage problem. Always confirm that
the bearing geometry used in the calculation matches the datasheet of the installed bearing.
