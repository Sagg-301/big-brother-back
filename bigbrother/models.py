from django.db import models
from django.contrib.auth.models import User as UserModel

# Create your models here.
# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models
from django.db.models.base import Model

class BaseModel(models.Model):
    id = models.AutoField(db_column="id",primary_key=True) # Field name made lowercase
    created_at = models.DateTimeField(db_column="created_at", auto_now_add=True)
    updated_at = models.DateTimeField(db_column="updated_at", auto_now=True)

    class Meta:
        abstract=True # Set this model as Abstract


class CrimesData(models.Model):
    id = models.TextField(db_column='ID', primary_key=True)
    case_number = models.TextField(db_column='Case Number', blank=True, null=True) 
    date = models.TextField(db_column='Date', blank=True, null=True) 
    block = models.TextField(db_column='Block', blank=True, null=True) 
    iucr = models.TextField(db_column='IUCR', blank=True, null=True) 
    primary_type = models.TextField(db_column='Primary Type', blank=True, null=True) 
    district = models.IntegerField(db_column='District', blank=True, null=True) 
    community_area = models.FloatField(db_column='Community Area', blank=True, null=True) 
    x_coordinate = models.FloatField(db_column='X Coordinate', blank=True, null=True) 
    y_coordinate = models.FloatField(db_column='Y Coordinate', blank=True, null=True) 

    class Meta:
        managed = False
        db_table = 'crimes_data'

class Prediction(BaseModel):
    x_coordinate = models.FloatField(db_column='x_coordinate', blank=True, null=True) 
    y_coordinate = models.FloatField(db_column='y_coordinate', blank=True, null=True)
    user = models.ForeignKey(UserModel,null= True, on_delete=models.SET_NULL)

    class Meta:
        db_table = 'location_prediction'