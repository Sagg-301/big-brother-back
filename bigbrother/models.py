from django.db import models

# Create your models here.
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class CrimesData(models.Model):
    id = models.TextField(db_column='ID', primary_key=True) # Field name made lowercase.
    case_number = models.TextField(db_column='Case Number', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    date = models.TextField(db_column='Date', blank=True, null=True)  # Field name made lowercase.
    block = models.TextField(db_column='Block', blank=True, null=True)  # Field name made lowercase.
    iucr = models.TextField(db_column='IUCR', blank=True, null=True)  # Field name made lowercase.
    primary_type = models.TextField(db_column='Primary Type', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    district = models.FloatField(db_column='District', blank=True, null=True)  # Field name made lowercase.
    community_area = models.FloatField(db_column='Community Area', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    x_coordinate = models.FloatField(db_column='X Coordinate', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.
    y_coordinate = models.FloatField(db_column='Y Coordinate', blank=True, null=True)  # Field name made lowercase. Field renamed to remove unsuitable characters.

    class Meta:
        managed = False
        db_table = 'crimes_data'