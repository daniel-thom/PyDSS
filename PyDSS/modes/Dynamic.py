from datetime import datetime, timedelta
from PyDSS.modes.abstract_solver import abstact_solver
import math

class Dynamic(abstact_solver):
    def __init__(self, dssInstance, SimulationSettings, Logger):
        super().__init__(dssInstance, SimulationSettings, Logger)
        print('Running Dynamic simulation')
        self.Settings = SimulationSettings
        self.pyLogger = Logger

        self._MaxItrs = SimulationSettings['Project']['Max Control Iterations']
        self._Time = datetime.strptime(SimulationSettings['Project']["Start time"], "%d/%m/%Y %H:%M:%S")
        self._StartTime = self._Time
        self._EndTime = self._Time + timedelta(minutes=SimulationSettings['Project']["Simulation duration (min)"])
        StartDay = (self._StartTime - datetime(self._StartTime.year, 1, 1)).days + 1
        StartTimeMin = self._StartTime.minute
        sStepResolution = SimulationSettings['Project']['Step resolution (sec)']

        self._sStepRes = sStepResolution
        self._dssIntance = dssInstance
        self._dssSolution = dssInstance.Solution

        self.StartDay = (self._StartTime - datetime(self._StartTime.year, 1, 1)).days + 1
        self.EndDay = (self._EndTime - datetime(self._EndTime.year, 1, 1)).days + 1

        self.setMode('Dynamic')
        self._dssSolution.Hour(StartDay * 24)
        self._dssSolution.Seconds(StartTimeMin * 60)
        self._dssSolution.Number(1)
        self._dssSolution.StepSize(self._sStepRes)
        self._dssSolution.MaxControlIterations(self._MaxItrs)
        return

    def setFrequency(self, frequency):
        self._dssSolution.Frequency(frequency)
        return

    def getFrequency(self):
        return  self._dssSolution.Frequency()

    def SimulationSteps(self):
        Seconds = (self._EndTime - self._StartTime).total_seconds()
        Steps = math.ceil(Seconds / self._sStepRes)
        return Steps, self._StartTime, self._EndTime

    def GetOpenDSSTime(self):
        return self._dssSolution.DblHour()

    def reset(self):
        self.setMode('Dynamic')
        self._dssSolution.Hour(self._Hour)
        self._dssSolution.Seconds(self._Second)
        self._dssSolution.Number(1)
        self._dssSolution.StepSize(self._sStepRes)
        self._dssSolution.MaxControlIterations(self._MaxItrs)
        return

    def SolveFor(self, mStartTime, mTimeStep):
        Hour = int(mStartTime/60)
        Min = mStartTime % 60
        self._dssSolution.Hour(Hour)
        self._dssSolution.Seconds(Min*60)
        self._dssSolution.Number(mTimeStep)
        self._dssSolution.Solve()
        return

    def IncStep(self):
        self._dssSolution.StepSize(self._sStepRes)
        self._dssSolution.Solve()
        self._Time = self._Time + timedelta(seconds=self._sStepRes)
        self._Hour = int(self._dssSolution.DblHour() // 1)
        self._Second = (self._dssSolution.DblHour() % 1) * 60 * 60
        self.pyLogger.debug('OpenDSS time [h] - ' + str(self._dssSolution.DblHour()))
        self.pyLogger.debug('PyDSS datetime - ' + str(self._Time))

    @property
    def MaxIterations(self):
        return self.Settings['Project']['Max Control Iterations']

    def GetTotalSeconds(self):
        return (self._Time - self._StartTime).total_seconds()

    def GetDateTime(self):
        return self._Time

    def GetStepResolutionSeconds(self):
        return self._sStepRes

    def GetStepSizeSec(self):
        return self._sStepRes

    def reSolve(self):
        self._dssSolution.StepSize(0)
        self._dssSolution.SolveNoControl()

    def Solve(self):
        self._dssSolution.StepSize(0)
        self._dssSolution.Solve()

    def getMode(self):
        return self._dssSolution.ModeID()

    def setMode(self, mode):
        self._dssIntance.utils.run_command('Set Mode={}'.format(mode))


