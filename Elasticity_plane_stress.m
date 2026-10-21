clear all                 % Clear all variables from the MATLAB workspace
close all                 % Close all open figure windows
clc                       % Clear the Command Window

%% 1. Define material properties
E = 30e6;                 % Young's modulus of the material
nu = 0.25;                % Poisson's ratio

%% 2. Define nodal coordinates
% Each row represents one node: [x-coordinate, y-coordinate]
%
% Node 1: (0, -1)
% Node 2: (2,  0)
% Node 3: (0,  1)

node = [0 -1;
        2  0;
        0  1];

%% 3. Define element connectivity
% The element contains nodes 1, 2 and 3.
% Each row of 'connect' represents one finite element.

connect = [1 2 3];

%% 4. Construct the constitutive matrix D for plane stress
% For an isotropic, linear-elastic material under plane stress:
%
%       {sigma_x}       [1   nu   0] {epsilon_x}
%       {sigma_y} = D = [nu  1    0] {epsilon_y}
%       {tau_xy}        [0   0  (1-nu)/2] {gamma_xy}
%
% D = E/(1-nu^2) * the matrix shown above.
%
% Stress vector: {sigma_x, sigma_y, tau_xy}^T
% Strain vector: {epsilon_x, epsilon_y, gamma_xy}^T

factor = E/(1-nu^2);

D = factor * [1       nu       0;
              nu      1        0;
              0       0  (1-nu)/2];

%% 5. Plot the original triangular element
% Plot nodal coordinates and connect them in the order specified.
% The markers 'o' indicate the nodes, and '-r' draws red edges.

figure;
plot(node(:,1), node(:,2), '-or', 'LineWidth', 1.5);
hold on;
axis equal;
grid on;
xlabel('x-coordinate');
ylabel('y-coordinate');
title('Three-node triangular finite element');

%% 6. Determine the number of nodes, elements and degrees of freedom
numnode = size(node,1);   % Number of nodes = 3
numelem = size(connect,1);% Number of elements = 1

ndof = 2;                 % Two displacement DOFs per node: u and v

% Total global degrees of freedom = number of nodes x DOFs per node
% Global displacement vector:
% U = [u1 v1 u2 v2 u3 v3]^T

K = zeros(numnode*ndof, numnode*ndof);
% Initialize the global stiffness matrix.
% For 3 nodes and 2 DOFs per node, K is a 6 x 6 matrix.

%% 7. Element stiffness matrix calculation
% The element stiffness matrix is evaluated using:
%
%       Ke = integral (B' * D * B) dA
%
% B = strain-displacement matrix
% D = material constitutive matrix
%
% Numerical integration is performed over the triangular element.

for iel = 1:numelem

    % Extract the global node numbers of the current element.
    conn = connect(iel,:);

    % Extract the coordinates of the element's three nodes.
    coord = node(conn,:);

    %% 7.1 Identify global degrees of freedom for this element
    % Each node has two DOFs:
    % Node 1 -> DOFs 1,2
    % Node 2 -> DOFs 3,4
    % Node 3 -> DOFs 5,6
    %
    % Hence, for this element:
    % gindex = [1 2 3 4 5 6]

    gindex = [];

    for i = 1:length(conn)
        gindex(2*i-1) = 2*conn(i)-1; % Horizontal displacement DOF
        gindex(2*i)   = 2*conn(i);   % Vertical displacement DOF
    end

    %% 7.2 Obtain quadrature points and weights
    % TRIANGULAR specifies the integration domain.
    % The last argument specifies the quadrature rule/order used
    % by the supplied quadrature function.
    %
    % W = integration weights
    % Q = coordinates of integration points in the reference triangle
    %
    % The exact meaning of the first argument depends on the
    % implementation of your quadrature function.

    [W,Q] = quadrature(1, 'TRIANGULAR', 2);

    %% 7.3 Loop over the integration points
    for igp = 1:size(W,1)

        % Coordinates of the current integration point
        % in the reference element.
        pt = Q(igp,:);

        % Evaluate shape functions and their natural-coordinate
        % derivatives at this integration point.
        %
        % T3 denotes a linear three-node triangular element.
        % N contains the shape functions.
        % dNdxi contains their derivatives with respect to
        % the natural coordinates.

        [N,dNdxi] = lagrange_basis('T3', pt);

        % Coordinates of the element nodes
        ncoord = node(conn,:);

        %% 7.4 Calculate the Jacobian matrix
        % Jo maps derivatives from natural coordinates to
        % physical coordinates (x,y).
        %
        % Jo = [dx/dxi  dy/dxi;
        %       dx/deta dy/deta]
        %
        % The exact arrangement follows the convention used by
        % lagrange_basis and the matrix multiplication below.

        Jo = dNdxi' * ncoord;

        % Physical coordinates of the integration point.
        % This is useful for plotting or evaluating fields.

        Gpt = N' * ncoord;

        % Display the integration point as a black square.
        plot(Gpt(1), Gpt(2), 'sk');

        %% 7.5 Transform derivatives into physical coordinates
        % The inverse Jacobian transforms natural derivatives
        % into derivatives with respect to x and y.
        %
        % dNdx contains:
        % Row 1 -> derivatives with respect to x
        % Row 2 -> derivatives with respect to y

        J = inv(Jo);

        dNdx = J * dNdxi';

        %% 7.6 Construct the strain-displacement matrix B
        % The small-strain vector is:
        %
        % {epsilon} = [epsilon_x; epsilon_y; gamma_xy]
        %
        % {epsilon} = B * {ue}
        %
        % where the element displacement vector is:
        % ue = [u1 v1 u2 v2 u3 v3]^T
        %
        % B is a 3 x 6 matrix for a T3 element.

        B = zeros(3,6);

        % Normal strain in the x-direction
        B(1,1:2:6) = dNdx(1,:);

        % Normal strain in the y-direction
        B(2,2:2:6) = dNdx(2,:);

        % Engineering shear strain gamma_xy
        B(3,1:2:6) = dNdx(2,:);
        B(3,2:2:6) = dNdx(1,:);

        %% 7.7 Calculate and assemble the element stiffness matrix
        % Differential area in the physical element:
        % dA = det(Jo) * d(xi) * d(eta)
        %
        % Element stiffness contribution:
        % dKe = B' * D * B * W * det(Jo)
        %
        % Contributions from all integration points are added.
        %
        % K(gindex,gindex) extracts the global matrix entries
        % corresponding to this element and adds the contribution.

        K(gindex,gindex) = K(gindex,gindex) ...
            + B' * D * B * W(igp) * det(Jo);

    end
end

%% 8. Define the global nodal displacement vector
% The displacement vector is arranged as:
%
% Uglobal = [u1 v1 u2 v2 u3 v3]
%
% Values are assigned in the same DOF order.
% These are prescribed/example displacements, not a solution
% obtained by solving the global equilibrium equations.

TempNode = [0.0 0.0025 0.0012 0 0.0 0.0025];

%% 9. Calculate strains and stresses in the element
% For each integration point:
%
% Strain = B * Ue
% Stress = D * Strain
%
% Ue is the displacement vector extracted for the current element.
% For a single-element model, Ue contains all six displacement DOFs.

for iel = 1:numelem

    % Get the nodes belonging to the element.
    conn = connect(iel,:);

    % Get the coordinates of the element nodes.
    coord = node(conn,:);

    %% 9.1 Extract the element displacement vector
    % The same DOF indexing used during stiffness assembly
    % is used to extract the displacement values.

    gindex = [];

    for i = 1:length(conn)
        gindex(2*i-1) = 2*conn(i)-1;
        gindex(2*i)   = 2*conn(i);
    end

    % Element displacement vector:
    % Ue = [u1 v1 u2 v2 u3 v3]^T

    udisp = TempNode(gindex);

    %% 9.2 Define quadrature points for stress recovery
    % The quadrature rule is used to specify where strain and
    % stress will be evaluated.
    %
    % For a linear T3 element with constant material properties
    % and small-strain linear elasticity, B is constant throughout
    % the element. Consequently, strain and stress are constant
    % within this element.

    [W,Q] = quadrature(2, 'TRIANGULAR', 2);

    %% 9.3 Evaluate strains and stresses
    for igp = 1:size(W,1)

        % Current integration point in natural coordinates.
        pt = Q(igp,:);

        % Shape functions and natural-coordinate derivatives.
        [N,dNdxi] = lagrange_basis('T3', pt);

        % Element nodal coordinates.
        ncoord = node(conn,:);

        % Jacobian matrix and physical location of the point.
        Jo = dNdxi' * ncoord;
        Gpt = N' * ncoord;

        % Plot the integration point.
        plot(Gpt(1), Gpt(2), 'sk');

        % Transform shape-function derivatives to x-y coordinates.
        J = inv(Jo);
        dNdx = J * dNdxi';

        % Construct the strain-displacement matrix.
        B = zeros(3,6);

        B(1,1:2:6) = dNdx(1,:);
        B(2,2:2:6) = dNdx(2,:);
        B(3,1:2:6) = dNdx(2,:);
        B(3,2:2:6) = dNdx(1,:);

        %% 9.4 Calculate the strain vector
        % Small-strain displacement relation:
        %
        % epsilon = B * Ue
        %
        % eps_e = [epsilon_x; epsilon_y; gamma_xy]

        eps_e = B * udisp';

        %% 9.5 Calculate the stress vector
        % Linear elastic constitutive relation:
        %
        % sigma = D * epsilon
        %
        % stress = [sigma_x; sigma_y; tau_xy]

        stress = D * eps_e;

    end
end

%% 10. Display the global stiffness matrix and final results
disp('Global stiffness matrix K:');
disp(K);

disp('Element displacement vector Ue:');
disp(udisp');

disp('Strain vector [epsilon_x; epsilon_y; gamma_xy]:');
disp(eps_e);

disp('Stress vector [sigma_x; sigma_y; tau_xy]:');
disp(stress);
