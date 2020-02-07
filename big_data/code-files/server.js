//command: node server.js

const express = require('express')
const bodyParser= require('body-parser')
const MongoClient = require('mongodb').MongoClient
const app = express()

app.use(bodyParser.urlencoded({extended: true}))

var db, result

MongoClient.connect('mongodb://192.168.1.21:27017', { useNewUrlParser: true }, (err, client) => {
  if (err) return console.log(err)
  db = client.db('tsla-stockdb')
  app.listen(3000, () => {
    console.log('Palvelu käynnistetty porttiin 3000')
  })
})

app.get('/', (req, res) => {
    db.collection('stockdata').find({},{Date:1, High:1, Low:1}).sort({Date: 1}).toArray((err, result) => {
        if (err) return console.log(err)
        res.render('index.ejs', {stockdata: result})
    })
})

app.get('/reload', (req, res) => {
    result = []
    db.collection('stockdata').find({},{Date:1, High:1, Low:1}).sort({Date: 1}).toArray((err, result) => {
        if (err) return console.log(err)
        res.json(result)
    })
})
