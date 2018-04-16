import pandas as pd
import networkx as nx
from math import sqrt

from bokeh.client import push_session
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.layouts import  column
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, BoxZoomTool, PanTool, WheelZoomTool, ResetTool, SaveTool

import matplotlib.colors as colors
import opendssdirect as dss
from opendssdirect.utils import run_command
import os

class dssNetworkX:
    #__Name = None
    #__Class = None
    def __init__(self, rootPath=os.getcwd(), SimSettings = None):
        self.__dssInstance = dss
        dssFiles = rootPath + '\\ProjectFiles\\' + SimSettings['Active Project'] + '\\DSSfiles'
        self.__dssFilePath = dssFiles + '\\' + SimSettings['DSS File']
        self.__dssInstance.Basic.ClearAll()
        self.__dssInstance.utils.run_command('Log=NO')
        run_command('Clear')
        run_command('compile ' + self.__dssFilePath)
        self.__dssGraph = nx.DiGraph()

        self.__CreateNodes()
        self.__CreatePDEdges()
        self.__CreatePCEdges()
        return

    def __CreateUniqueNodePropertyDict(self, Property):
        i = 0
        Dict = {}
        for Node in self.__dssGraph.nodes():
            if Property in self.__dssGraph.node[Node]:
                PptyValue = self.__dssGraph.node[Node][Property]
                if PptyValue not in Dict:
                    Dict[PptyValue] = i
                    i += 1
            else:
                print(Property + ' is not a valid property for nodes')
                break
        return Dict

    def __CreateUniqueLinePropertyDict(self, Property):
        i = 0
        Dict = {}
        for Line in self.__dssGraph.edges():
            Node1 = list(Line)[0]
            Node2 = list(Line)[1]
            if Property in self.__dssGraph[Node1][Node2]:
                PptyValue = self.__dssGraph[Node1][Node2][Property]
                if PptyValue not in Dict:
                    Dict[PptyValue] = i
                    i += 1
            else:
                print(Property + ' is not a valid property for edges')
                break
        return Dict

    def GetUniqueEdgeProperties(self):
        EdgeAttrList = []
        for Edge in self.__dssGraph.edges():
            Node1 = list(Edge)[0]
            Node2 = list(Edge)[1]
            for Attr in self.__dssGraph[Node1][Node2].keys():
                if Attr not in EdgeAttrList:
                    EdgeAttrList.append(Attr)
        return EdgeAttrList

    def GetUniqueNodeProperties(self):
        NodeAttrList = []
        for Node in self.__dssGraph.nodes():
            for Attr in self.__dssGraph.node[Node].keys():
                NodeAttrList.append(Attr)
            break
        return NodeAttrList

    def CreateGraphVisualization(self, Settings = None):
        Col1 = 120
        Col2 = 10
        Layouts = {
            'Circular': nx.circular_layout,
            'Spring': nx.spring_layout,
            'Fruchterman': nx.fruchterman_reingold_layout,
            'Spectral': nx.spectral_layout,
            'Random': nx.random_layout,
            'Shell': nx.shell_layout,
        }

        NodeColorProperty = Settings['NodeColorProperty']
        LineColorProperty = Settings['LineColorProperty']
        Iterations = Settings['Iterations']
        Layout = Settings['Layout']
        ShowRefNode = Settings['ShowRefNode']
        NodeSize = Settings['NodeSize']
        OpenBrowser = Settings['Open plots in browser']

        NodeDict = self.__CreateUniqueNodePropertyDict(NodeColorProperty)
        LineDict = self.__CreateUniqueLinePropertyDict(LineColorProperty)
        ColorList = list(colors.cnames.keys())

        NodeColor = [ColorList[Col2 + NodeDict[i[1][NodeColorProperty]]] for i in self.__dssGraph.nodes(data=True)]
        self.__NodeList = list(NodeDict.keys())
        self.__LineList = list(LineDict.keys())
        self.__NodeColor = [ColorList[Col2 + x] for x in range(len(self.__NodeList))]
        self.__LineColor = [ColorList[Col1 + x] for x in range(len(self.__LineList))]
        try:
            layout = Layouts[Layout](self.__dssGraph, iterations=Iterations)
        except:
            layout = Layouts[Layout](self.__dssGraph)#, k = 1.1/sqrt(self.__dssGraph.number_of_nodes()), iterations= 50)
        NodeData =  pd.DataFrame([[i[0],
                                   layout[i[0]][0],
                                   layout[i[0]][1],
                                   i[1]['kVBase'],
                                   i[1]['TotalMiles'],
                                   i[1]['NumNodes'],
                                   i[1]['N_Customers'],
                                   i[1]['Distance'],
                                   i[1]['ConnectedPDs'],
                                   i[1]['ConnectedPCs']] for i in self.__dssGraph.nodes(data=True)]
                                 , columns=['Name',
                                            'X',
                                            'Y',
                                            'kVBase',
                                            'TotalMiles',
                                            'NumNodes',
                                            'N_Customers',
                                            'Distance',
                                            'ConnectedPDs',
                                            'ConnectedPCs',]).set_index('Name')
        self.DataSource = ColumnDataSource(NodeData)
        hoverBus = HoverTool(tooltips=[
            ('Name','@Name'),
            ("(x,y)", "(@X, @Y)"),
            ('kVBase','@kVBase'),
            ('TotalMiles','@TotalMiles'),
            ('NumNodes','@NumNodes'),
            ('N_Customers','@N_Customers'),
            ('Distance','@Distance'),
            ('Connected PDs', '@ConnectedPDs'),
            ('Connected PCs', '@ConnectedPCs'),
        ])
        Xs = []
        Ys = []
        LineColors = []

        for Edge in self.__dssGraph.edges():
            Node1 = list(Edge)[0]
            Node2 = list(Edge)[1]
            if ShowRefNode:
                Xs.append([NodeData.loc[Node1]['X'], NodeData.loc[Node2]['X']])
                Ys.append([NodeData.loc[Node1]['Y'], NodeData.loc[Node2]['Y']])
                LineColors.append(ColorList[Col1 + LineDict[self.__dssGraph[Node1][Node2][LineColorProperty]]])
            else:
                if Node2 != 'Ref Ground Node':
                    Xs.append([NodeData.loc[Node1]['X'], NodeData.loc[Node2]['X']])
                    Ys.append([NodeData.loc[Node1]['Y'], NodeData.loc[Node2]['Y']])
                    LineColors.append(ColorList[Col1 + LineDict[self.__dssGraph[Node1][Node2][LineColorProperty]]])
        plot = figure(tools=[ResetTool(), hoverBus, BoxSelectTool(), SaveTool(),
                             BoxZoomTool(), WheelZoomTool(), PanTool()],
                      title="Distribution network graph representation",
                      plot_width=650,
                      plot_height=600,
                      )
        C = plot.circle('X', 'Y', source=self.DataSource, size=NodeSize, level='overlay', color=NodeColor, legend='Nodes')
        L = plot.multi_line(Xs, Ys, color=LineColors, legend='Edges')
        plot.legend.location = "top_left"
        plot.legend.click_policy = "hide"
        #output_file("networkx_graph.html")
        doc = curdoc()
        doc.add_root(plot)
        doc.title = "PyDSS"
        self.__session = push_session(doc)
        if OpenBrowser:
            show(plot)
        return

    def GetColorList(self):
        A2 = zip(self.__LineList, self.__LineColor)
        A1 = zip(self.__NodeList, self.__NodeColor)
        return A1, A2

    def GetSessionID(self):
        return self.__session.id

    def __CreatePCEdges(self):
        PCElement = self.__dssInstance.Circuit.FirstPCElement()
        while PCElement:
            ElementData = {
                'Name'             : self.__dssInstance.CktElement.Name().split('.')[1],
                'Class'            : self.__dssInstance.CktElement.Name().split('.')[0],
                'BusFrom'          : self.__dssInstance.CktElement.BusNames()[0].split('.')[0],
                'PhasesFrom'       : self.__dssInstance.CktElement.BusNames()[0].split('.')[1:],
                'BusTo'            : 'Ref Ground Node',
            }
            self.__dssGraph.node[ElementData['BusFrom']]['ConnectedPCs'] += ' ' + ElementData['Class']
            ElementData = self.__ExtendPCElementDict(ElementData)
            self.__dssGraph.add_edge(ElementData['BusFrom'], ElementData['BusTo'])
            self.__dssGraph[ElementData['BusFrom']][ElementData['BusTo']] = ElementData
            PCElement = self.__dssInstance.Circuit.NextPCElement()
        return

    def __ExtendPCElementDict(self, Dict):
        #print(self.__dssInstance.CktElement.AllPropertyNames())
        NumberOfProperties = len(self.__dssInstance.Element.AllPropertyNames())
        for i in range(NumberOfProperties):
            try:
                PropertyName = self.__dssInstance.Properties.Name(str(i))
                X = self.__dssInstance.Properties.Value(str(i))
                if X is not None and X is not '' and PropertyName is not None and PropertyName is not '':
                    Dict[PropertyName] = X
            except:
                pass
        return Dict

    def __CreatePDEdges(self):
        PDElement = self.__dssInstance.Circuit.FirstPDElement()
        while PDElement:
            ElementData = {
                'Name'             : self.__dssInstance.CktElement.Name().split('.')[1],
                'Class'            : self.__dssInstance.CktElement.Name().split('.')[0],
                'BusFrom'          : self.__dssInstance.CktElement.BusNames()[0].split('.')[0],
                'PhasesFrom'       : self.__dssInstance.CktElement.BusNames()[0].split('.')[1:],
                'BusTo'            : self.__dssInstance.CktElement.BusNames()[1].split('.')[0],
                'PhasesTo'         : self.__dssInstance.CktElement.BusNames()[1].split('.')[1:],
                'Enabled'          : self.__dssInstance.CktElement.Enabled(),
                'HasSwitchControl' : self.__dssInstance.CktElement.HasSwitchControl(),
                'HasVoltControl'   : self.__dssInstance.CktElement.HasVoltControl(),
                'IsOpen'           : self.__dssInstance.CktElement.IsOpen(),
                'GUID'             : self.__dssInstance.CktElement.GUID(),
                'NumConductors'    : self.__dssInstance.CktElement.NumConductors(),
                'NumControls'      : self.__dssInstance.CktElement.NumControls(),
                'NumPhases'        : self.__dssInstance.CktElement.NumPhases(),
                'NumTerminals'     : self.__dssInstance.CktElement.NumTerminals(),
                'OCPDevType'       : self.__dssInstance.CktElement.NumTerminals(),
                'IsShunt'          : self.__dssInstance.PDElements.IsShunt(),
                'NumCustomers'     : self.__dssInstance.PDElements.NumCustomers(),
                'ParentPDElement'  : self.__dssInstance.PDElements.ParentPDElement(),
                'SectionID'        : self.__dssInstance.PDElements.SectionID(),
                'TotalCustomers'   : self.__dssInstance.PDElements.TotalCustomers(),
                'TotalMiles'       : self.__dssInstance.PDElements.TotalMiles(),
            }
            self.__dssGraph.node[ElementData['BusFrom']]['ConnectedPDs'] += ' ' + ElementData['Class']
            self.__dssGraph.node[ElementData['BusTo']]['ConnectedPDs'] += ' ' + ElementData['Class']
            self.__dssGraph.add_edge(ElementData['BusFrom'], ElementData['BusTo'])
            self.__dssGraph[ElementData['BusFrom']][ElementData['BusTo']] = ElementData
            PDElement = self.__dssInstance.Circuit.NextPDElement()
        return

    def __CreateNodes(self):
        Buses = self.__dssInstance.Circuit.AllBusNames()
        BusData = {}
        for Bus in Buses:
            self.__dssInstance.Circuit.SetActiveBus(Bus)
            BusData = {
                'Name'          : self.__dssInstance.Bus.Name(),
                'X'             : self.__dssInstance.Bus.X(),
                'Y'             : self.__dssInstance.Bus.Y(),
                'kVBase'        : self.__dssInstance.Bus.kVBase(),
                'Zsc0'          : self.__dssInstance.Bus.Zsc0(),
                'Zsc1'          : self.__dssInstance.Bus.Zsc1(),
                'TotalMiles'    : self.__dssInstance.Bus.TotalMiles(),
                'SectionID'     : self.__dssInstance.Bus.SectionID(),
                'Nodes'         : self.__dssInstance.Bus.Nodes(),
                'NumNodes'      : self.__dssInstance.Bus.NumNodes(),
                'N_Customers'   : self.__dssInstance.Bus.N_Customers(),
                'Distance'      : self.__dssInstance.Bus.Distance(),
                'ConnectedPDs'  : '',
                'ConnectedPCs'  : '',
            }
            self.__dssGraph.add_node(self.__dssInstance.Bus.Name())
            self.__dssGraph.node[self.__dssInstance.Bus.Name()] = BusData
        self.__dssGraph.add_node('Ref Ground Node')
        BusData['X'] = 0
        BusData['Y'] = 0
        BusData['kVBase'] = 0
        BusData['TotalMiles'] = 0
        BusData['NumNodes'] = 0
        BusData['N_Customers'] = 0
        BusData['Distance'] = 0
        self.__dssGraph.node['Ref Ground Node'] = BusData
        return

SS = {
        'Active Project'         : 'HECO',
        'DSS File'               : 'MasterCircuit_Mikilua_baseline2.dss',

    }

AA = {
        'Layout'                 : 'Circular',
        'Iterations'             : 100,
        'ShowRefNode'            : False,
        'NodeSize'               : 10,
        'LineColorProperty'      : 'Class',
        'NodeColorProperty'      : 'ConnectedPCs',
        'Open plots in browser'  : True,
}

# if __name__ == '__main__':
#     A = dssNetworkX(SimSettings=SS)
#     A.CreateGraphVisualization(Settings=AA)