# implementations of Point and Vector classes

import numbers
import numpy as np
import scipy as sc

# Point class implementation
# --------------------------
class Point:

    def __init__(self, *coords):
        """
        Инициализация
        """
        self.coords = np.array(coords)
        self.dim = len(self.coords)


    def __repr__(self):
        """
        Вывод координат
        """
        coord_str = ''
        for coord in self.coords:
            coord_str += '{:.6f}'.format(coord) + ', '
        out = 'Point' + ' ' + '(' + coord_str + ')'
        return out
    

    def __add__(self, other):
        """
        Сложение точки и вектора
        """
        warn = 'Only the addition of instances of the Vector'\
            ' and Point classes is available'
        # Если складываемый объект не является
        # вектором, то вызывается ошибка
        if not isinstance(other, Vector):
            raise ValueError(warn)
        # Точка складывается с вектором,
        # результатом является новая точка
        else:
            res = self.coords + other.coords
            return Point(*res.tolist())
        

    def __sub__(self, other):
        """
        Вычитание вектора из точки или 
        точки из точки
        """
        warn = 'Only the subtraction of instances of the Point'\
            ' and Vector of Point and Point classes is available'
        # Если вычитаемый объект не является
        # вектором или точкой, то вызывается ошибка
        if not (isinstance(other, Vector) | isinstance(other, Point)):
            raise ValueError(warn)
        # Из точки вычитается точка,
        # результатом является новый вектор
        elif isinstance(other, Point):
            res = self.coords - other.coords
            return Vector(*res.tolist())
        # Из точки вычитается вектор,
        # результатом является новая точка
        elif isinstance(other, Vector):
            res = self.coords - other.coords
            return Point(*res.tolist())
        

    def projection(self, curve, *args):
        """
        Проецирование точки на касательную к кривой
        """
        # Две точки касательной для построения вектора
        t0, dt = args
        tangent_sp = Point(*curve.TangLine(t0).value(t0))
        tangent_ep = Point(*curve.TangLine(t0).value(t0+dt))
        
        # Вектор от проецируемой точки до точки касания
        vector1 = self - tangent_sp
        # Вектор касательной
        vector2 = tangent_sp - tangent_ep

        scalar_product = (vector1 * vector2) / np.linalg.norm(vector2)**2
        res = tangent_sp + scalar_product * vector2
        return res
    

    def obj(self, t, curve):
        """
        Целевая функция для поиска параметра,
        соответствующего точке проекции
        """
        dt = 1
        line = curve.TangLine(t)
        # q_ = self.projection(line, t, dt)
        v1 = self - Point(*line.value(t))
        v2 = Point(*line.value(t)) - Point(*line.value(t+dt))
        k = 1
        out = k * (v1 * v2)**2
        return np.sum(out)
    

    def projection_loop(self, t0, curve):
        """
        Возвращает параметр точки проекции,
        найденный с помощью решения задачи
        минимизации целевой функции
        """
        func = lambda t: self.obj(t, curve=curve)
        res = sc.optimize.minimize(fun=func, x0=t0)
        return res.x[0]
    

# Vector class implementation
# ---------------------------
class Vector:

    def __init__(self, *coords):
        """
        Инициализация
        """
        self.coords = np.array(coords)
        self.dim = len(self.coords)


    def __repr__(self):
        """
        Вывод информации об экземпляре класса
        """
        coord_str = ''
        for coord in self.coords:
            coord_str += '{:.6f}'.format(coord) + ', '
        out = 'Vector' + ' ' + '(' + coord_str + ')'
        return out
    

    def __add__(self, other):
        """
        Сложение вектора и вектора или
        вектора и точки
        """
        warn = 'Only the addition of instances of the Vector'\
            ' and Point or Vector and Vector classes is available'
        
        # Если складываемый объект не является точкой
        # или вектором, то вызывается ошибка
        if not (isinstance(other, Vector) | isinstance(other, Point)):
            raise ValueError(warn)
        # Если вектор складывается с точкой,
        # то результатом является новая точка
        elif isinstance(other, Point):
            res = self.coords + other.coords
            return Point(*res.tolist())
        # Если вектор складывается с вектором,
        # то результатом является новый вектор
        elif isinstance(other, Vector):
            res = self.coords + other.coords
            return Vector(*res.tolist())
        
    
    def __sub__(self, other):
        """
        Вычитание вектора и вектора
        """
        warn = 'Only the subtraction of instances of the Vector'\
            ' and Vector classes is available'
        
        # Если вычитаемый объект не является вектором
        # то вызывается ошибка
        if not isinstance(other, Vector):
            raise ValueError(warn)
        # Если из вектора вычитается вектор,
        # то результатом является новый вектор
        else:
            res = self.coords - other.coords
            return Vector(*res.tolist())
        
    
    def __mul__(self, other):
        """
        Скалярное умножение
        """
        warn = 'Only the scalar product of instances of the Vector'\
            ' and Vector classes or instance of the Vector сlass and\
                 Scalar is available'
        
        # Умножение вектора на вектор
        if isinstance(other, Vector) :
            return np.dot(self.coords, other.coords)
        # Умножение вектора на число
        elif isinstance(other, numbers.Number):
            res = self.coords * other
            return Vector(*res.tolist())
        # 
        else:
            raise ValueError(warn)
        
    
    def __rmul__(self, other):
        """
        Отражение операции скалярного умножения
        """
        return self.__mul__(other)
    

    def __matmul__(self, other):
        """
        Векторное умножение векторов
        """
        warn = 'Only the inner product of instances of the Vector'\
            ' and Vector classes is available'
        
        # Если объект не является вектором
        # вызывается исключение
        if not isinstance(other, Vector):
            raise ValueError(warn)
        else:
            res = np.cross(self.coords, other.coords)
            return Vector(*res.tolist())
        
        
    def __setitem__(self, j, value):
        """
        Задает значение j-ой координате вектора
        """
        self.coords[j] = value
        
    
    def normalize(self):
        """
        Возвращает вектор сонаправленный с единичной нормой,
        сонаправленный данному вектору
        """
        norm = np.linalg.norm(self.coords)
        res = self.coords / norm
        return Vector(*res.tolist())
    

# Parametric curve class implementation
# -------------------------------------

class ParametricCurve:

    def __init__(self, *equations):
        self.x, self.y, self.dxdt, self.dydt = equations
    

    def value(self, t):
        """
        Возвращает координаты кривой, соотвествующие
        значению параметра t
        """
        return np.array([self.x(t),
                         self.y(t)])
    

    def deriv(self, t):
        """
        Возвращает значения производной кривой,
        соотвествующие значению параметра t
        """
        return np.array([self.dxdt(t),
                         self.dydt(t)])
    
    
    def TangLine(self, t0):
        """
        Возвращает параметрическое представление
        касательной линии в точке, соответствующей
        значению параметра t0

        Параметры
        ---------
        t0 : number
            параметр точки касания
        """
        n_dim = len(self.value(t0))
        n_eq = 2 * n_dim
        coeffs = np.empty(shape=n_eq, dtype=object)

        for i in range(n_dim):
            val = self.value(t0)[i]
            dval = self.deriv(t0)[i]
            coeffs[i] = val
            coeffs[i+n_dim] = dval

        x = lambda t: coeffs[0] + coeffs[0+n_dim] * (t - t0)
        y = lambda t: coeffs[1] + coeffs[1+n_dim] * (t - t0)
        dxdt = lambda t: coeffs[0+n_dim] * t
        dydt = lambda t: coeffs[1+n_dim] * t
        equations = [x, y, dxdt, dydt]
        return ParametricCurve(*equations)