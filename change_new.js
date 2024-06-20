// use traps 

// var temp = db.cessations.findOne({_id: 'ND6dTLAZm2Ct8SYJd'});
// db.statewides.insert({_id: "ND6dTLAZm2Ct8SYJd"}, {$set: temp}, {})



// var tempCursor = db.cessation_announcements.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_announcements.insert(doc);
// });


// tempCursor = db.cessation_resources.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// db.statewide_resources.insertMany(tempCursor)
// tempCursor.forEach((doc) => {
//   db.statewide_resources.insert(doc);
// });


// tempCursor = db.cessation_operations.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_operations.insert(doc)
// });
// tempCursor = db.cessation_activity_types.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_activity_types.insert(doc)
// });
// tempCursor = db.cessation_grantees.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_grantees.insert(doc)
// });
// tempCursor = db.cfs.cessationOpFiles.filerecord.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.cfs_gridfs.statewideOpFiles.insert(doc)
// });
// tempCursor = db.cessation_operation_completions.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_operation_completions.insert(doc)
// });

// tempCursor = db.cessation_activities.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_activities.insert(doc)
// });
// tempCursor = db.cessation_geo_locations.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_geo_locations.insert(doc)
// });
// tempCursor = db.cfs_gridfs.cessationResourceFiles.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.cfs_gridfs.statewideResourceFiles.insert(doc)
// });
// tempCursor = db.cessation_audiences.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_audiences.insert(doc)
// });
// tempCursor = db.cessation_calendar.find({group_id: 'ND6dTLAZm2Ct8SYJd'});
// tempCursor.forEach((doc) => {
//   db.statewide_calendar.insert(doc)
// });



// db.cessations.update({_id: 'ND6dTLAZm2Ct8SYJd'},  {$set: {archived: true}}, {multi: true})
// db.cessation_announcements.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'},  {$set: {archived: true}}, {multi: true})
// db.cessation_resources.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'},  {$set: {archived: true}}, {multi: true})
// db.cessation_operations.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'},  {$set: {archived: true}}, {multi: true})
// db.cessation_activity_types.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'},  {$set: {archived: true}}, {multi: true})
// db.cessation_grantees.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});
// db.cfs.cessation_op_files.filerecord.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});
// db.cessation_operation_completions.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});

// db.cessation_activities.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});
// db.cessation_geo_locations.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});
// db.cfs_gridfs.cessationResourceFiles.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});

// db.cessation_audiences.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});
// db.cessation_calendar.updateMany({group_id: 'ND6dTLAZm2Ct8SYJd'}, {$set: {archived: true}}, {multi: true});