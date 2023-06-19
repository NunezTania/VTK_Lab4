#!/usr/bin/env python
#
# Labo 4
# Le but de ce laboratoire est de visualiser le parcours d'un planeur 
# décollant depuis le lac gelé de Ottsjön en Suède. Le vol de ce planeur 
# nous permettra par ailleurs de visualiser les flux d'air ascendant et 
# descendant sous l'influence du relief des montagnes environnantes. 
#
# Auteurs: Mélissa Gehring et Tania Nunez 

import vtk
import numpy as np
from pyproj import Transformer
import math
import pandas as pd
from vtk import *

# ====================== Constantes ======================

# Relevé GPS du planeur
GPS_FILE = "data/vtkgps.txt"

# Carte de la région
MAP_FILE = "data/glider_map.jpg"
# Coordonnées RT90 des coins de la carte
TOP_LEFT_RT90 = (1349349, 7022573)
TOP_RIGHT_RT90 = (1371573, 7022967)
BOTTOM_RIGHT_RT90 = (1371835, 7006362)
BOTTOM_LEFT_RT90 = (1349602, 7005969)
# Coordonnées WGS84 des coins de la carte
TOP_LEFT_WGS84 = Transformer.from_crs("EPSG:3021", "EPSG:4326").transform(TOP_LEFT_RT90[1], TOP_LEFT_RT90[0])
TOP_RIGHT_WGS84 = Transformer.from_crs("EPSG:3021", "EPSG:4326").transform(TOP_RIGHT_RT90[1], TOP_RIGHT_RT90[0])
BOTTOM_RIGHT_WGS84 = Transformer.from_crs("EPSG:3021", "EPSG:4326").transform(BOTTOM_RIGHT_RT90[1], BOTTOM_RIGHT_RT90[0])
BOTTOM_LEFT_WGS84 = Transformer.from_crs("EPSG:3021", "EPSG:4326").transform(BOTTOM_LEFT_RT90[1], BOTTOM_LEFT_RT90[0])
COORDS_WGS84 = np.array([BOTTOM_LEFT_WGS84, BOTTOM_RIGHT_WGS84, TOP_RIGHT_WGS84, TOP_LEFT_WGS84])

# Modèle d'élévation de la région
ELEVATION_FILE = "data/EarthEnv-DEM90_N60E010.bil"
# Bornes du fichier
LON_MIN = 10.0
LON_MAX = 15.0
LAT_MIN = 60.0
LAT_MAX = 65.0
# Nombre de points en longitude et latitude
NUM_WIDTH = 6000
NUM_HEIGHT = 6000

# Rayon de la terre
EARTH_RAD = 6371009

# Rayon des lignes de coupe
LINE_RADIUS = 50

# Rayon de la trajectoire du glider
GLIDER_LINE_RADIUS = 25

# Colors
BKG_COLOR = (1, 1, 1)
TEXT_COLOR = (0., 0., 0.1)
LINE_COLOR = (0., 0.5, 0.5)
GLIDER_LINE_COLOR_RANGE = (-5, 5)

# Taille de la fenêtre
WIN_WIDTH = 1200
WIN_HEIGHT = 800

# ====================== Classes ======================

# Interactor pour la souris
class MouseInteractor(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, map, text, *args, **kwargs):
        # Modifier la carte en fonction du déplacement de la souris
        self.AddObserver("MouseMoveEvent", self.mouse_move_event)

        # Acteurs
        self.map = map
        self.text = text

        # Picker pour récupérer les coordonnées du point sous la souris
        self.picker = vtk.vtkPointPicker()
        self.picker.PickFromListOn()
        self.picker.AddPickList(self.map)

        # Sphère correspondant à la terre
        self.earth = vtk.vtkSphere()

        # Cutter pour couper le modèle d'élévation
        self.cutter = vtk.vtkCutter()
        self.cutter.SetCutFunction(self.earth)
        self.cutter.SetInputData(self.map.GetMapper().GetInput())

        # Stripper pour transformer les coupes en lignes
        self.stripper = vtk.vtkStripper()
        self.stripper.SetInputConnection(self.cutter.GetOutputPort())

        # Tube pour donner du volume aux lignes
        self.tube = vtk.vtkTubeFilter()
        self.tube.SetRadius(LINE_RADIUS)
        self.tube.SetInputConnection(self.stripper.GetOutputPort())

        # Mapper pour les lignes
        self.line_mapper = vtk.vtkPolyDataMapper()
        self.line_mapper.ScalarVisibilityOff()
        self.line_mapper.SetInputConnection(self.tube.GetOutputPort())

        # Acteur pour les lignes
        self.line_actor = vtk.vtkActor()
        self.line_actor.GetProperty().SetColor(LINE_COLOR)
        self.line_actor.SetMapper(self.line_mapper)

    def lateInit(self):
        # Ajouter les lignes à la scène
        self.GetDefaultRenderer().AddActor(self.line_actor)

    def mouse_move_event(self, obj, event):
        # Récupére la position de la souris
        pos = self.GetInteractor().GetEventPosition()

        # Récupère les coordonnées du point sous la souris
        self.picker.Pick(pos[0], pos[1], 0, self.GetDefaultRenderer())
        
        # Vérifier que le point est bien sur la carte
        if self.picker.GetActor() == self.map:
            # Calcule le rayon de la sphère en fonction de la position de la souris
            rad = np.linalg.norm(self.picker.GetPickPosition())
            self.earth.SetRadius(rad)

            # Mettre à jour le cutter
            self.cutter.Update()

            # Mettre à jour le texte
            altitude = self.picker.GetDataSet().GetPointData().GetScalars().GetValue(self.picker.GetPointId())
            self.text.SetInput("Altitude: {:.0f} m".format(altitude))

            self.GetInteractor().Render()

        # Met à jour le display
        self.GetInteractor().Render()
        self.OnMouseMove()

# ====================== Fonctions ======================

def toCartesian(lat, lon, alt):
    transform = vtk.vtkTransform()
    transform.RotateX(lon)
    transform.RotateY(-lat)
    transform.Translate(0, 0, alt)
    return transform.TransformPoint(0, 0, 0)

def quadInterpolation(x, y, a, b):
    # Source : https://www.particleincell.com/2012/quad-interpolation/
    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]

    # m = (-b+sqrt(b^2-4ac))/(2a)
    det = math.sqrt(bb ** 2 - 4 * aa * cc)
    m = (-bb - det) / (2 * aa)

    l = (x - a[0] - a[2] * m) / (a[1] + a[3] * m)

    return l, m

def getImplicitBoolean():
    # Créer un clipper pour couper le modèle d'élévation avec 4 plans
    implicitBool = vtk.vtkImplicitBoolean()
    implicitBool.SetOperationTypeToUnion()

    point0 = np.array([0, 0, 0])
    point1 = np.array(toCartesian(*COORDS_WGS84[0], 5))
    point2 = np.array(toCartesian(*COORDS_WGS84[1], 5))
    point3 = np.array(toCartesian(*COORDS_WGS84[2], 5))
    point4 = np.array(toCartesian(*COORDS_WGS84[3], 5))
    # Créer un plan pour chaque côté du quadrilatère
    for p in [(point1, point2), (point2, point3), (point3, point4), (point4, point1)]:
        norm = -np.cross(p[0] - point0, p[1] - point0)
        plane = vtk.vtkPlane()
        plane.SetNormal(norm)
        implicitBool.AddFunction(plane)
    return implicitBool

def createMapActor():
    # Source : https://www.particleincell.com/2012/quad-interpolation/
    # Matrice pour l'interpolation quadrilaterale
    interpolation_matrix = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]])

    # Multiplication matricielle pour les alphas and betas pour l'interpolation quadrilaterale
    alphas = interpolation_matrix.dot(COORDS_WGS84[:, 0])
    betas = interpolation_matrix.dot(COORDS_WGS84[:, 1])
    
    # Points à interpoler
    longitudes, latitudes = np.meshgrid(
        np.linspace(LON_MIN, LON_MAX, NUM_WIDTH),
        np.linspace(LAT_MIN, LAT_MAX, NUM_HEIGHT))
    latitudes, longitudes = latitudes.flatten(), longitudes.flatten()

    # Masques pour filtrer les points à interpoler
    lat_min_mask = latitudes >= COORDS_WGS84[:, 0].min()
    lat_max_mask = latitudes <= COORDS_WGS84[:, 0].max()
    lon_min_mask = longitudes >= COORDS_WGS84[:, 1].min()
    lon_max_mask = longitudes <= COORDS_WGS84[:, 1].max()
    lat_mask = lat_min_mask & lat_max_mask
    lon_mask = lon_min_mask & lon_max_mask
    all_mask = lat_mask & lon_mask 

    # Nombre de lignes et de colonnes à interpoler
    rows = latitudes[lat_mask].size // NUM_WIDTH
    cols = longitudes[lon_mask].size // NUM_HEIGHT

    # Filtrer les points à interpoler en fonction des masques
    altitudes = np.fromfile(ELEVATION_FILE, dtype=np.int16)
    altitudes = altitudes[all_mask]
    latitudes = latitudes[all_mask]
    longitudes = longitudes[all_mask]

    # Création des points, de leur altitude et des textures
    points = vtk.vtkPoints()
    pointsAlt = vtk.vtkIntArray()
    pointsTexture = vtk.vtkFloatArray()
    pointsTexture.SetNumberOfComponents(2)
    for latitude, longitude, altitude in zip(latitudes, longitudes, altitudes):
        point = toCartesian(latitude, longitude, altitude + EARTH_RAD)
        points.InsertNextPoint(point)
        pointsAlt.InsertNextValue(altitude)
        pointsTexture.InsertNextTuple(quadInterpolation(latitude, longitude, alphas, betas))

    # Création de la carte
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(cols, rows, 1)
    grid.SetPoints(points)
    grid.GetPointData().SetScalars(pointsAlt)
    grid.GetPointData().SetTCoords(pointsTexture)

    # Création du mapper et du terrain clippé
    clipped = vtk.vtkClipDataSet()
    clipped.SetInputData(grid)
    clipped.SetClipFunction(getImplicitBoolean())
    clipped.Update()
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(clipped.GetOutputPort())
    mapper.ScalarVisibilityOff()

    # Création de la texture
    texture = vtk.vtkTexture()
    texture.RepeatOff()
    imageReader = vtk.vtkJPEGReader()
    imageReader.SetFileName(MAP_FILE)
    texture.SetInputConnection(imageReader.GetOutputPort())

    # Création de l'acteur
    map = vtk.vtkActor()
    map.SetMapper(mapper)
    map.SetTexture(texture)

    return map


def createGliderActor():
    # Lecture des données GPS
    gliderData = pd.read_table(GPS_FILE, header=None, skiprows=[0],
                         delim_whitespace=True, dtype={1: np.int32, 2: np.int32, 3: np.float64})
    # Calcul de la différences de hauteurs entre les points consécutifs
    gliderData[4] = gliderData[3].diff()

    # Process des données GPS
    gliderData[1], gliderData[2] = Transformer.from_crs("EPSG:3021", "EPSG:4326").transform(gliderData[2], gliderData[1])
    gliderData[5] = gliderData.apply(lambda x: toCartesian(x[1], x[2], x[3] + EARTH_RAD), axis=1)
    altitudes, coords =  gliderData[4], gliderData[5]

    # Génère les points et les altitudes
    points = vtk.vtkPoints()
    pointsAlt = vtk.vtkFloatArray()

    for coord, alt in zip(coords, altitudes):
        points.InsertNextPoint(coord)
        pointsAlt.InsertNextValue(alt)

    # Générer le tube
    line = vtk.vtkLineSource()
    line.SetPoints(points)
    line.Update()

    lineData = line.GetOutput()
    lineData.GetPointData().SetScalars(pointsAlt)

    tube = vtk.vtkTubeFilter()
    tube.SetRadius(GLIDER_LINE_RADIUS)
    tube.SetInputConnection(line.GetOutputPort())

    # Générer le mapper et l'acteur
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tube.GetOutputPort())
    mapper.SetScalarRange(GLIDER_LINE_COLOR_RANGE)

    output = vtk.vtkActor()
    output.SetMapper(mapper)

    return output

def createTextActor():
    text = vtk.vtkTextActor()
    text.SetInput("")
    text.GetTextProperty().SetColor(TEXT_COLOR)
    text.GetTextProperty().SetBackgroundColor(BKG_COLOR)
    text.GetTextProperty().SetBackgroundOpacity(1)
    text.GetTextProperty().SetFontSize(30)
    text.SetPosition((50, 50))

    return text

if __name__ == '__main__':
    # Création des acteurs
    map = createMapActor()
    glider = createGliderActor()
    text = createTextActor()

    # Création de la scène et ajout des acteurs
    ren = vtk.vtkRenderer()
    ren.SetBackground(BKG_COLOR)
    ren.AddActor(map)
    ren.AddActor(glider)
    ren.AddActor(text)
    # Création de la fenêtre
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    renWin.SetSize(WIN_WIDTH, WIN_HEIGHT)

    # Création de l'interacteur
    interactor = MouseInteractor(map, text)
    interactor.SetDefaultRenderer(ren)
    interactor.lateInit()
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    iren.SetInteractorStyle(interactor)

    # Lancement de l'application
    iren.Initialize()
    renWin.Render()
    iren.Start()