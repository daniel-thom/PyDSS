from PyDSS.dssBus import dssBus

class dssCircuit:


    def __init__(self, dssInstance):
        self.__Name = None
        self.__Variables = {}
        self.__Name =  dssInstance.Circuit.Name()
        self.__dssInstance = dssInstance

        CktElmVarDict = dssInstance.Circuit.__dict__
        for key in CktElmVarDict.keys():
            try:
                self.__Variables[key] = getattr(dssInstance.Circuit, key)
            except:
                self.__Variables[key] = None
                pass
        return

    def GetInfo(self):
        return self.__Name

    def inVariableDict(self,VarName):
        if VarName in self.__Variables:
            return True
        else:
            return False

    def DataLength(self,VarName):
        if VarName in self.__Variables:
            VarValue = self.GetVariable(VarName)
        else:
            return 0, None
        if  isinstance(VarValue, list):
            return len(VarValue) , 'List'
        elif isinstance(VarValue, str):
            return 1, 'String'
        elif isinstance(VarValue, int or float or bool):
            return 1, 'Number'
        else:
            return 0, None

    def IsValidAttribute(self,VarName):
        if VarName in self.__Variables:
            return True
        else:
            return False

    def GetValue(self,VarName):
        if VarName in self.__Variables:
            VarValue = self.GetVariable(VarName)
        else:
            VarValue = -1
        return VarValue

    def GetVariable(self,VarName):
        if VarName in self.__Variables:
            try:
                return self.__Variables[VarName]()
            except:
                print ('Unexpected error')
                return None
        else:
            print (VarName + ' is an invalid variable name for element ' + self.__Name)
            return None

    # all below added by npanossi
    def SetParameter(self, Param, Value):
        self.__dssInstance.utils.run_command(self.__Name + '.' + Param + ' = ' + str(Value))
        print('dssInstance Circuit ' + Param + ' set to ' + str(Value))
        return None
        #self.GetParameter(Param)

    # def GetParameter(self, Param):
    #         self.__dssInstance.Circuit.SetActiveElement(self.__Class + '.' + self.__Name)
    #         if self.__dssInstance.Element.Name() == (self.__Class + '.' + self.__Name):
    #             x = self.__dssInstance.Properties.Value(Param)
    #             try:
    #                 return float(x)
    #             except:
    #                 return x
    #         else:
    #             print('Could not set ' + self.__Class + '.' + self.__Name + ' as active element.')
    #             return None