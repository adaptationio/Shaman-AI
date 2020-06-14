import numpy as np
import random
from .auth import Auth
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from sklearn import preprocessing
import csv
#import torch
import time
import cv2
import mss
import numpy
import re
import datetime
from .pairs import PairsConfigured


# THERE IS A LOT OF CODE NOT USED HERE AND I NEED TO GET RID OF, EG THE PIARS IN THE INIT 

pc = PairsConfigured()
class DataGrabber():
    """gets data and processes ready to use"""

    def __init__(self, config, eval=False):
        self.pc = pc
        self.eval = eval
        self.config = config
        self.love = 14
        self.auth = Auth(self.config, self.eval)
        self.client = oandapyV20.API(access_token=self.auth.access_token)
        self.usd_crypto = ['BTC_USD']
        self.usd_pairs = ['AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CAD', 'USD_CNY', 'USD_CHF', 'NZD_USD']
        self.usd_commod = ['BCO_USD', 'XAU_USD', 'XAG_USD', 'WTICO_USD', 'CORN_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'SUGAR_USD', 'SOYBN_USD', 'NATGAS_USD', 'XCU_USD' ]
        self.usd_index = ['US30_USD', 'SPX500_USD', 'NAS100_USD', 'US2000_USD' ]
        self.usd_bonds = ['USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD']
        self.aud_pairs = ['AUD_USD','AUS_JPY', 'AUD_NZD', 'EUR_AUD', 'AUD_CAD', 'GBP_AUD', 'AUD_CHF', 'AUD_CNY']
        self.aud_index = ['AU200_AUD']
        self.aud_commod = ['XAG_AUD', 'XAU_AUD']
        self.aud_chf = ['AUD_CHF', 'AUD_USD', 'USD_CHF', 'AUD_JPY', 'AUD_NZD', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'CAD_JPY', 'AUD_CAD' , 'CAD_CHF', 'GBP_AUD', 'NZD_CHF', 'XAG_AUD', 'XAG_CHF', 'XAU_AUD', 'XAU_CHF', 'AU200_AUD']
        self.aud_usd_all = ['AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'BCO_USD', 'XAU_USD', 'XAG_USD', 'WTICO_USD', 'CORN_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'SUGAR_USD', 'SOYBN_USD', 'NATGAS_USD', 'XCU_USD', 'US30_USD', 'SPX500_USD', 'NAS100_USD', 'US2000_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'AUD_JPY', 'AUD_NZD', 'EUR_AUD', 'AUD_CAD', 'GBP_AUD', 'AUD_CHF', 'AU200_AUD', 'XAG_AUD', 'XAU_AUD']
        self.aud_usd_less = ['AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CAD', 'USD_CHF', 'NZD_USD', 'US30_USD', 'SPX500_USD', 'NAS100_USD', 'US2000_USD', 'AUD_JPY', 'AUD_NZD', 'EUR_AUD', 'AUD_CAD', 'GBP_AUD', 'AUD_CHF', 'AU200_AUD']
        self.gbp_cad = ['GBP_CAD', 'GBP_USD', 'USD_CAD', 'EUR_GBP','EUR_CAD', 'GBP_CHF', 'GBP_JPY', 'CAD_JPY', 'UK100_GBP', 'AUD_CAD' , 'CAD_CHF', 'GBP_AUD',  'GBP_NZD', 'NZD_CAD', 'XAG_CAD', 'XAG_GBP', 'XAU_CAD', 'XAU_GBP']
        self.years = ['2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008','2007', '2006', '2005', '2004', '2003', '2002', '2001', '2000', '1999', '1998', '1997']
        self.instrument = ['EUR_USD', 'AUD_USD', 'GBP_USD', 'NZD_USD', 'USD_CHF', 'USD_CAD']
        self.time = ['00:00:00']
        self.aud_cad = ['AUD_CAD', 'AUD_USD', 'AUD_JPY', 'AUD_NZD', 'USD_CAD', 'EUR_CAD', 'EUR_AUD', 'CAD_JPY', 'CAD_CHF', 'GBP_AUD', 'GBP_CAD', 'NZD_CAD', 'AUD_CHF', 'XAG_AUD','XAG_CAD', 'XAU_AUD', 'XAU_CAD', 'AU200_AUD']
        self.aud_nzd = ['AUD_NZD', 'AUD_USD', 'AUD_JPY', 'EUR_AUD', 'NZD_JPY', 'NZD_USD', 'AUD_CAD' , 'EUR_NZD', 'GBP_AUD', 'GBP_NZD', 'NZD_CAD', 'AUD_CHF', 'NZD_CHF', 'XAG_AUD', 'XAG_NZD', 'XAU_AUD', 'XAU_NZD', 'AU200_AUD']
        self.gbp_jpy = ['GBP_JPY', 'AUD_USD', 'USD_JPY', 'GBP_USD', 'AUD_JPY', 'EUR_GBP','EUR_JPY', 'GBP_CHF', 'CHF_JPY', 'CAD_JPY', 'NZD_JPY', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'XAG_JPY', 'XAU_JPY', 'XAG_GBP', 'XAU_GBP']
        self.all = ['AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 'AUD_JPY', 'AUD_NZD', 'USD_CAD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'GBP_CHF', 'GBP_JPY', 'CHF_JPY', 'BCO_USD', 'DE30_EUR', 'US30_USD','XAU_USD', 'CAD_JPY', 'HK33_HKD', 'JP225_USD', 'NZD_JPY', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'AUD_CAD' , 'AUD_SGD', 'CAD_CHF', 'CHF_ZAR', 'EUR_CZK', 'EUR_DKK', 'EUR_HUF', 'EUR_NOK','EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_TRY', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'GBP_ZAR', 'NZD_CAD','TRY_JPY', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_NOK', 'USD_PLN','USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'XAG_JPY', 'XAU_JPY', 'ZAR_JPY','CAD_SGD', 'EUR_ZAR', 'GBP_PLN', 'GBP_SGD', 'NZD_SGD', 'SGD_JPY', 'AUD_CHF', 'NZD_CHF', 'SGD_CHF', 'AUD_HKD', 'EUR_HKD', 'GBP_HKD', 'CAD_HKD', 'CHF_HKD', 'HKD_JPY', 'NZD_HKD', 'SGD_HKD', 'XAG_AUD','XAG_CAD', 'XAG_CHF', 'XAG_EUR', 'XAG_GBP', 'XAG_HKD', 'XAG_NZD', 'XAG_SGD', 'XAU_AUD', 'XAU_CAD', 'XAU_CHF', 'XAU_EUR', 'XAU_GBP', 'XAU_HKD', 'XAU_NZD', 'XAU_SGD', 'XAU_XAG', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'NL25_EUR', 'EU50_EUR', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'FR40_EUR', 'DE10YB_EUR', 'US2000_USD', 'AU200_AUD', 'SG30_SGD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.eur_gbp = [ 'EUR_GBP', 'EUR_USD', 'GBP_USD', 'EUR_CHF','EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'GBP_CHF', 'GBP_JPY', 'DE30_EUR', 'UK100_GBP', 'EUR_NZD', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'XAG_EUR', 'XAG_GBP', 'XAU_EUR', 'XAU_GBP', 'NL25_EUR', 'EU50_EUR', 'FR40_EUR', 'DE10YB_EUR']
        self.gbp_chf = ['GBP_CHF', 'GBP_USD', 'USD_CHF', 'EUR_CHF', 'EUR_GBP', 'GBP_JPY', 'CHF_JPY', 'UK100_GBP', 'CAD_CHF', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD',  'AUD_CHF', 'NZD_CHF',  'XAG_CHF', 'XAG_GBP', 'XAU_CHF', 'XAU_GBP']
        #self.gbp_chf = ['AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 'AUD_JPY', 'AUD_NZD', 'USD_CAD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'GBP_CHF', 'GBP_JPY', 'CHF_JPY', 'BCO_USD', 'DE30_EUR', 'US30_USD','XAU_USD', 'CAD_JPY', 'HK33_HKD', 'JP225_USD', 'NZD_JPY', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'AUD_CAD' , 'AUD_SGD', 'CAD_CHF', 'CHF_ZAR', 'EUR_CZK', 'EUR_DKK', 'EUR_HUF', 'EUR_NOK','EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_TRY', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'GBP_ZAR', 'NZD_CAD','TRY_JPY', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_NOK', 'USD_PLN','USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'XAG_JPY', 'XAU_JPY', 'ZAR_JPY','CAD_SGD', 'EUR_ZAR', 'GBP_PLN', 'GBP_SGD', 'NZD_SGD', 'SGD_JPY', 'AUD_CHF', 'NZD_CHF', 'SGD_CHF', 'AUD_HKD', 'EUR_HKD', 'GBP_HKD', 'CAD_HKD', 'CHF_HKD', 'HKD_JPY', 'NZD_HKD', 'SGD_HKD', 'XAG_AUD','XAG_CAD', 'XAG_CHF', 'XAG_EUR', 'XAG_GBP', 'XAG_HKD', 'XAG_NZD', 'XAG_SGD', 'XAU_AUD', 'XAU_CAD', 'XAU_CHF', 'XAU_EUR', 'XAU_GBP', 'XAU_HKD', 'XAU_NZD', 'XAU_SGD', 'XAU_XAG', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'NL25_EUR', 'EU50_EUR', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'FR40_EUR', 'DE10YB_EUR', 'US2000_USD', 'AU200_AUD', 'SG30_SGD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.gbp_nzd = ['GBP_NZD', 'GBP_USD', 'AUD_NZD', 'EUR_GBP', 'GBP_CHF', 'GBP_JPY', 'NZD_JPY', 'NZD_USD', 'UK100_GBP', 'EUR_NZD', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'NZD_CAD',  'XAG_GBP', 'XAG_NZD', 'XAU_GBP', 'XAU_NZD']
        self.eur_usd_all = ['EUR_USD', 'AUD_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 'USD_CAD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_CAD', 'EUR_AUD',  'DE30_EUR', 'US30_USD','XAU_USD', 'JP225_USD', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'EUR_CZK', 'EUR_DKK', 'EUR_HUF', 'EUR_NOK','EUR_NZD', 'EUR_PLN', 'EUR_SEK', 'EUR_TRY', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_NOK', 'USD_PLN','USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'EUR_ZAR', 'EUR_HKD', 'XAG_EUR', 'XAU_EUR', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'NL25_EUR', 'EU50_EUR', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'FR40_EUR', 'DE10YB_EUR', 'US2000_USD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.gbp_usd_small = ['GBP_USD','AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF',  'USD_CAD', 'EUR_GBP', 'GBP_CHF', 'GBP_JPY', 'US30_USD','XAU_USD', 'JP225_USD', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'XAG_GBP', 'XAU_GBP', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD','US2000_USD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.eur_usd_less = ['EUR_USD', 'AUD_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF', 'USD_CAD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_CAD', 'EUR_AUD',  'DE30_EUR', 'US30_USD','XAU_USD', 'JP225_USD', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'EUR_NZD', 'XAG_EUR', 'XAU_EUR', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'NL25_EUR', 'EU50_EUR', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'FR40_EUR', 'DE10YB_EUR', 'US2000_USD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.usd_chf = ['USD_CHF','AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD',  'USD_CAD', 'EUR_CHF', 'GBP_CHF', 'CHF_JPY', 'US30_USD','XAU_USD', 'JP225_USD','XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD','CAD_CHF', 'CHF_ZAR', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_NOK', 'USD_PLN','USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR','AUD_CHF', 'NZD_CHF', 'SGD_CHF', 'XAG_CHF', 'XAU_CHF', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'US2000_USD','USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.usd_jpy = ['USD_JPY','AUD_USD','EUR_USD', 'GBP_USD', 'USD_CHF', 'AUD_JPY', 'USD_CAD', 'EUR_JPY', 'GBP_JPY', 'CHF_JPY', 'US30_USD','XAU_USD', 'CAD_JPY', 'JP225_USD', 'NZD_JPY', 'NZD_USD', 'XAG_USD', 'SPX500_USD', 'WTICO_USD', 'XAG_JPY', 'XAU_JPY', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD',  'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD', 'US2000_USD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.hour = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
        self.minute = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32','33','34', '35', '36', '37', '38','39','40','41','42','43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59']
        self.day_feb = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28']
        self.day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
        self.day_30 = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
        self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
        self.granularity= ['M1', 'M5', 'M15', 'M30', 'H1', 'H4']
        self.gbp_usd = ['GBP_USD','AUD_USD','EUR_USD', 'USD_JPY', 'GBP_USD', 'USD_CHF',  'USD_CAD', 'EUR_GBP', 'GBP_CHF', 'GBP_JPY', 'US30_USD','XAU_USD', 'JP225_USD', 'NZD_USD', 'XAG_USD', 'UK100_GBP', 'SPX500_USD', 'WTICO_USD', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'GBP_ZAR', 'USD_CZK', 'USD_DKK', 'USD_HKD', 'USD_HUF', 'USD_INR', 'USD_MXN', 'USD_NOK', 'USD_PLN','USD_SAR', 'USD_SEK', 'USD_SGD', 'USD_THB', 'USD_TRY', 'USD_ZAR', 'GBP_PLN', 'GBP_SGD', 'GBP_HKD', 'XAG_GBP', 'XAU_GBP', 'CORN_USD', 'NAS100_USD', 'WHEAT_USD', 'XPD_USD', 'XPT_USD', 'SOYBN_USD', 'NATGAS_USD','XCU_USD', 'USB02Y_USD', 'USB05Y_USD', 'USB10Y_USD', 'USB30Y_USD','US2000_USD', 'USD_CNH', 'CN50_USD', 'IN50_USD', 'TWIX_USD']
        self.gbp_aud = ['GBP_AUD', 'AUD_USD', 'GBP_USD', 'AUD_JPY', 'AUD_NZD', 'EUR_GBP', 'EUR_AUD', 'GBP_CHF', 'GBP_JPY', 'UK100_GBP', 'AUD_CAD' , 'GBP_CAD', 'GBP_NZD', 'AUD_CHF', 'XAG_AUD', 'XAG_GBP', 'XAU_AUD', 'XAU_GBP', 'AU200_AUD']
        self.eur_aud = ['EUR_AUD', 'AUD_USD','EUR_USD', 'AUD_JPY', 'AUD_NZD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_CAD', 'DE30_EUR', 'AUD_CAD' , 'EUR_NZD', 'GBP_CAD', 'AUD_CHF', 'XAG_AUD', 'XAG_EUR', 'XAU_AUD', 'XAU_CHF', 'XAU_EUR', 'NL25_EUR', 'EU50_EUR', 'AU200_AUD']
        
        self.eur_cad = ['EUR_CAD', 'AUD_USD','EUR_USD', 'USD_CAD', 'EUR_CHF', 'EUR_GBP','EUR_JPY', 'EUR_AUD', 'DE30_EUR', 'CAD_JPY', 'AUD_CAD' , 'CAD_CHF', 'EUR_NZD', 'GBP_CAD', 'NZD_CAD', 'XAG_CAD', 'XAG_EUR', 'XAU_CAD', 'XAU_EUR', 'NL25_EUR', 'EU50_EUR']

        self.eur_gbp = ['EUR_GBP', 'AUD_USD','EUR_USD', 'GBP_USD', 'EUR_CHF', 'EUR_JPY', 'EUR_CAD', 'EUR_AUD', 'GBP_CHF', 'GBP_JPY', 'DE30_EUR', 'UK100_GBP', 'EUR_NZD', 'GBP_AUD', 'GBP_CAD', 'GBP_NZD', 'XAG_EUR', 'XAG_GBP', 'XAU_EUR', 'XAU_GBP', 'NL25_EUR', 'EU50_EUR']
        self.config = config
        self.day_feb_2 = ['28']
        self.month31 = []
        if self.eval == True:
            self.years_list = [ '2018', '2005', '2017']
        else:
            self.years_list = ['2016', '2015', '2014', '2013', '2012', '2011', '2010', '2008', '2007', '2006',]
        self.instrument_list = ['EUR_USD']
        
        #dodgy 2011, '2009' '2008'
        #self.years_list = ['2017']
        
        self.year = random.choice(self.years_list)
        self.instruments = random.choice(self.instrument_list)
        #print(self.year)
        
        #self.full_year = np.load('data/AUD_USD2190H42017.npy')
        
        

    def get_candles(self, _from,  count, granularity, instrument):
        params = {"from": _from, "count": count, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        data = self.client.request(r)
        #print(data)
        return data

    def data_converted(self, data):
        data_converted  = []
        for i in data['candles']:
            if i['complete'] == True:
                data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume'], i['time']])

        return data_converted


    def normalize(self, x):
        normalized = preprocessing.normalize(x)
        return normalized


    def scaled(self, x):
        scaled = preprocessing.scale(x)
        return scaled

    
    def tocsv(self, x, path):
        with open(path, "a", newline='') as fp:
            wr = csv.writer(fp, dialect='excel')
            for i in range(len(x)):
                wr.writerow(x[i])

    def totensor(self, data):
        data = torch.from_numpy(data)
        return data

    def batcher(self, x, y, batch_size):
        x_data = list()
        return x_data

    def toarray(self, x):
        x = np.array(x, dtype=np.float32)
        return x

    def process_to_normalized(self):
        year = random.choice(self.years)
        data = self.get_candles(year+'-01-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        return data

    def process_to_array(self):
        instrument = 'EUR_USD'
        year = random.choice(self.years)
        day = random.choice(self.day)
        month = random.choice(self.month)
        hour = random.choice(self.hour)
        minute = random.choice(self.minute)
        day = random.choice(self.day)
        data = self.get_candles(year+'-'+month+'-'+day+'T00:00:00Z', 4320, "M1", instrument)
        data = self.data_converted(data)
        data = self.time_to_array(data)
        data = self.toarray(data)
    
        return data

    def process_to_array_2(self):
        
        self.years = ['2018']
        #self.month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']
        self.hist = 365
        #self.hist = 5760
        self.t_frame = "D"
        #self.month = ['01','02','3']
        #self.years = ['2019']
        self.instrument = ['SPX500_USD']
        #self.instrument = ['AUD_USD']
        for i in self.instrument:
            for y in self.years:
                full_data = []
                for m in self.month:
                    if m == "02":
                        day = self.day_feb
                    if m == "04" or m == '06' or m == '08' or m == '09':
                        day = self.day_30
                    else:
                        day = self.day
                    for d in day:
                        print(i)
                        data = self.get_candles(y+'-'+m+'-'+d+'T21:00:00Z', self.hist, self.t_frame, i)
                        data = self.data_converted(data)
                        data = self.time_to_array(data)
                        data = self.toarray(data)
            
                        full_data.append(data)
        #full_data = self.flatten_simple(full_data)
                np.save('data/'+ self.instrument[0] + str(self.hist) + self.t_frame + y + '.npy', full_data)

        return full_data

    def process_to_array_full(self):
        
        
        self.years = [ '2018', '2017','2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005']
        self.hist = 2190
        #self.years = ['2019']
        self.t_frame = "H4"
        self.instrument = self.config.pair
        #self.instruments_all = self.eur_usd_less
        #self.instruments_all = self.usd_chf
        self.instruments_all= self.config.pair_list
        for y in self.years:
            full_data = []
            for i in self.instruments_all:
                #print(i)
                data = self.get_candles(y+'-01-01T00:00:00Z', self.hist, self.t_frame, i)
                data = self.data_converted(data)
                data = self.time_to_array(data)
                data = self.toarray(data)

                full_data.append(data)
        #full_data = self.flatten_simple(full_data)
            np.save('data/'+'less'+ self.instruments_all[0] + str(self.hist) + self.t_frame + y + '.npy', full_data)

        return full_data

    def eval_results(self, data):
        try:
            results = np.load('data/'+self.config.pair+self.config.year+'.npy')
            results = list(results)
        except:
            results = []
            #results = self.toarray(data)
        results.append(data)
        np.save('data/'+self.config.pair+self.config.year+'.npy', results)
        print('results:')
        print(results)
        print(np.max(results))
        print(np.argmax(results))
        return results
    
    def best_model(self, config):
        try:
            results_2018 = np.load('data/'+self.config.pair+'2018'+'.npy')
        except:
            return print('2018 missing')
        try:
            results_2019 = np.load('data/'+self.config.pair+'2019'+'.npy')
        except:
            return print('2019 missing')
        results_all = []
        #best = 0
        best_agent = 0
        for i in range(len(results_2019)):
            data_2018 = results_2018[i]
            data_2019 = results_2019[i]
            both = data_2018 + data_2019
            both = both / 2
            results_all.append(both)
            best_agent = np.argmax(results_all)
            best_results = np.max(results_all)
        
        print('Best Agent:')
        print(best_agent)
        print('Best Result:')
        print(best_results)
        print(results_2018[int(best_agent)])
        print(results_2019[int(best_agent)])
        return best_agent
            
                

            
        
        
        #np.save('data/'+self.config.pair+'best'+'.npy', results_all)

        


        


    def process_to_tensor(self):
        data = self.get_candles('2016-06-01T00:00:00Z', 2880, "M1", "EUR_USD")
        data = self.data_converted(data)
        data = self.toarray(data)
        data = self.normalize(data)
        data = self.totensor(data)
        return data

    def flatten_full(self, markeallt, user):
        market = data

        x = list()
        for i in range(len(old_data)):
            con = np.concatenate((data), axis=None)
            con = np.concatenate((con, old_data[i][1]), axis=None)
            con[0].tolist()
            x.append(con)
        return x

    def flatten(self, u, m):
        u = np.concatenate((u), axis=None)
        m = np.concatenate((m), axis=None)
        #c = np.concatenate((c), axis=None)
        flattened = np.concatenate((m, u), axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened
        
    def flatten2(self, u):
        u = np.concatenate((u), axis=None)
    
        flattened = np.concatenate(u, axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened
    def flatten_simple(self, u):
        u = np.concatenate((u), axis=None)
        #m = np.concatenate((m), axis=None)
        #c = np.concatenate((c), axis=None)
        flattened = np.concatenate((u), axis=None)

        #k = self.data_grabber.flatten(market_details, player_details)
        return flattened

    def get_screen(self):
        with mss.mss() as sct:
            # Part of the screen to capture
            monitor = {"top": 40, "left": 0, "width": 800, "height": 640}

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct.grab(monitor))

                # Display the picture
                #cv2.imshow("OpenCV/Numpy normal", img)

                # Display the picture in grayscale
                # cv2.imshow('OpenCV/Numpy grayscale',
                #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))

                print("fps: {}".format(1 / (time.time() - last_time)))

                # Press "q" to quit
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
    def difference(self, state):
        new_state = []
        r = 194
        for i in range(96):
            before = state[i][0]
            b = i+1
            after = state[b][0]
            diff = after - before
            vol = state[b][4]
            o = state[b][3]
            l = state[b][2]
            h = state[b][1]

            new_state.append([after, diff, vol, o, l, h ])
        return new_state
    
    def difference2(self, state):
        #data_converted.append([i['mid']['c'], i['mid']['h'], i['mid']['l'], i['mid']['o'], i['volume']])
        new_state = []
        r = 194
        for i in range(194):
            c = state[i][0]
            h = state[i][1]
            l = state[i][2]
            o = state[i][3]
            v = state[i][4]
            c = c - o
            h = h - o
            l = l - o
            

            new_state.append([c, h, l, v])
        return new_state


    def load_state(self, arg, arg2):
        #self.eval = True
        if self.eval == True:
            self.year =arg2
        else:
            self.years_list = ['2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008', '2007', '2006', '2005']
        
            self.year = random.choice(self.years_list)
        self.instruments=arg
        print(self.year)
        self.full_year = np.load('data/'+'less'+self.instruments+'2190H4'+self.year+'.npy', allow_pickle=True)
        day = self.full_year
        #day = random.choice(self.full_year)
        #day = self.full_year[arg]
        return day
    
    def time_to_array(self, data):
        for i in range(len(data)):
            date = data[i][5]
            date = re.split('-|T|:|Z|', date)
            date = date[0:5]
            date = list(map(int, date))
            day = datetime.date(date[0], date[1], date[2]).weekday()
            hour = date[3]
            minute = date[4]
            date = [day, hour, minute]
            data[i][5] = day
            data[i].append(hour)
            data[i].append(minute)
        return data

    

    
#dates = ["2016", "2017", "2018"]
#test = DataGrabber()
#test.process_to_array_2()
#data = test.load_state_2()
#print(len(test.full_year[0]))
#print(len(data[1]))
#candles = test.get_candles('1998-06-01T00:00:00Z', 1, "M15", "EUR_USD")
#print(candles)
#some_data = test.data_converted(candles)
#some_data = test.toarray(some_data)
#some_data = test.normalize(some_data)
#some_data = test.totensor(some_data)
#data_day = some_data[0:1440]
#print(len(data_day))
#print(len(some_data))
#print(candles)
#print(some_data)
#test.get_screen()
#state = [1,2,3,4,5,6,7,8,9,10]
#statenew = state[-4:]
#print(statenew )

