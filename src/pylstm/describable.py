#!/usr/bin/python
# coding=utf-8
from __future__ import division, print_function, unicode_literals
import numpy as np
import sys
from copy import deepcopy


class Describable(object):
    """
    Base class for all objects that can be described and initialized from a
    description.
    """
    __undescribed__ = {}

    @classmethod
    def __get_all_undescribed__(cls):
        ignore = {}
        for c in reversed(cls.__mro__):
            if hasattr(c, '__undescribed__'):
                c_ignore = c.__undescribed__
                if isinstance(c_ignore, dict):
                    ignore.update(c_ignore)
                elif isinstance(c_ignore, set):
                    ignore.update({k: None for k in c_ignore})
        return ignore

    def __describe__(self):
        """
        Returns a description of this object. That is a dictionary
        containing the name of the class as '$type' and all members of the
        class. This description is json-serializable.

        If a sub-class of Describable contains non-describable members, it has
        to override this method to specify how it should be described.

        :rtype: dict
        """
        description = {}
        ignorelist = self.__get_all_undescribed__()
        for member, value in self.__dict__.items():
            if member in ignorelist:
                continue
            description[member] = get_description(value)

        description['$type'] = self.__class__.__name__
        return description

    def __init_from_description__(self, description):
        """
        Initializes this object from a given description.

        If a sub-class of Describable contains non-describable fields, it has to
        override this method to specify how they should be initialized from
        their description.

        :param description: description of this object
        :type description: dict
        """
        assert self.__class__.__name__ == description['$type'], \
            "Description for '%s' has wrong type '%s'" % (
                self.__class__.__name__, description['$type'])

        for member, default_val in self.__get_all_undescribed__().items():
            self.__dict__[member] = deepcopy(default_val)

        for member, descr in description.items():
            if member == '$type':
                continue
            self.__dict__[member] = create_object_from_description(descr)


def get_description(this):
    if isinstance(this, Describable):
        return this.__describe__()
    elif isinstance(this, list):
        return [get_description(v) for v in this]
    elif isinstance(this, np.ndarray):
        return this.tolist()
    elif isinstance(this, dict):
        return {k: get_description(v) for k, v in this.items()}
    elif (isinstance(this, bool) or
          isinstance(this, numerical_types) or
          isinstance(this, text_type)):
        return this
    else:
        raise TypeError('Type: "%s" is not describable' % type(this))


def create_object_from_description(description):
    if isinstance(description, dict):
        if '$type' in description:
            name = description['$type']
            for describable in _get_inheritors(Describable):
                if describable.__name__ == name:
                    instance = describable.__new__(describable)
                    instance.__init_from_description__(description)
                    return instance
            raise ValueError('No describable class "%s" found!' % name)
        else:
            return {k: create_object_from_description(v)
                    for k, v in description.items()}
    elif (isinstance(description, bool) or
          isinstance(description, numerical_types) or
          isinstance(description, text_type)):
        return description
    elif isinstance(description, list):
        return [get_description(d) for d in description]

    return None


def _get_inheritors(cls):
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


if sys.version < '3':
    numerical_types = (int, long, float)
    text_type = basestring
else:
    numerical_types = (int, float)
    text_type = str